"""Node optimizer for optimizing module configurations."""

import gc
import itertools as it
import json
import logging
import os
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.trial import Trial

from autointent import Dataset
from autointent.context import Context
from autointent.custom_types import NodeType, SearchSpaceValidationMode
from autointent.nodes.info import NODES_INFO
from autointent.schemas.node_validation import ParamSpaceFloat, ParamSpaceInt, ParamSpaceT, SearchSpaceConfig

logger = logging.getLogger(__name__)


class NodeOptimizer:
    """Class for optimizing nodes in a computational pipeline.

    This class is responsible for optimizing different modules within a node
    using various search strategies and logging the results.
    """

    def __init__(
        self,
        node_type: NodeType,
        search_space: list[dict[str, Any]],
        target_metric: str,
        metrics: list[str] | None = None,
    ) -> None:
        """Initializes the node optimizer.

        Args:
            node_type: The type of node being optimized.
            search_space: A list of dictionaries defining the search space.
            target_metric: The primary metric to optimize.
            metrics: Additional metrics to track during optimization.
        """
        self._logger = logger
        self.node_type = node_type
        self.node_info = NODES_INFO[node_type]
        self.target_metric = target_metric

        self.metrics = metrics if metrics is not None else []
        if self.target_metric not in self.metrics:
            self.metrics.append(self.target_metric)

        self.validate_search_space(search_space)
        self.modules_search_spaces = search_space

    def fit(
        self,
        context: Context,
    ) -> None:
        """Performs the optimization process for the node.

        Args:
            context: The optimization context containing relevant data.

        Raises:
            AssertionError: If an invalid sampler type is provided.
        """
        self._logger.info("Starting %s node optimization...", self.node_info.node_type.value)

        # TODO use node specific hpo_config
        if context.hpo_config.sampler == "tpe":
            sampler_instance = optuna.samplers.TPESampler(
                seed=context.seed,
                consider_prior=context.hpo_config.consider_prior,
                prior_weight=context.hpo_config.prior_weight,
                n_startup_trials=context.hpo_config.n_startup_trials,
                n_ei_candidates=context.hpo_config.n_ei_candidates,
                constant_liar=context.hpo_config.constant_liar,
            )
        elif context.hpo_config.sampler == "random":
            sampler_instance = optuna.samplers.RandomSampler(seed=context.seed)  # type: ignore[assignment]

        study, finished_trials, n_trials = load_or_create_study(
            study_name=self.node_info.node_type,
            context=context,
            direction="maximize",
            sampler=sampler_instance,
            n_trials=context.hpo_config.n_trials,
        )
        self._counter = finished_trials  # zero if study is newly created

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        obj = partial(self.objective, search_space=self.modules_search_spaces, context=context)

        study.optimize(
            obj,
            n_trials=n_trials,
            n_jobs=context.hpo_config.n_jobs,
            gc_after_trial=True,
            timeout=context.hpo_config.timeout,
        )

        self._logger.info("%s node optimization is finished!", self.node_info.node_type)

    def objective(
        self,
        trial: Trial,
        search_space: list[dict[str, Any]],
        context: Context,
    ) -> float:
        """Defines the objective function for optimization.

        Args:
            trial: The Optuna trial instance.
            module_name: The name of the module being optimized.
            search_space: The parameter search space.
            context: The execution context.

        Returns:
            The value of the target metric for the given trial.
        """
        module_name, module_hyperparams = self._suggest_module_and_hyperparams(trial, search_space)

        if prev_metric := _check_duplicate(trial):
            msg = f"Duplicated trial with {module_name=}, {prev_metric=}, {module_hyperparams=}"
            logger.debug(msg)
            return prev_metric

        self._logger.debug("Initializing %s module with config: %s", module_name, json.dumps(module_hyperparams))
        module = self.node_info.modules_available[module_name].from_context(context, **module_hyperparams)
        module_hyperparams.update(module.get_implicit_initialization_params())

        context.callback_handler.start_module(
            module_name=module.trial_name,
            num=self._counter,
            module_kwargs=module_hyperparams,
        )

        self._logger.debug("Scoring %s module...", module_name)

        quality_metrics = module.score(context, metrics=self.metrics)

        target_metric = quality_metrics[self.target_metric]
        all_metrics = context.callback_handler.update_metrics(quality_metrics)
        context.callback_handler.log_metrics(all_metrics)
        context.callback_handler.end_module()

        context.optimization_info.log_module_optimization(
            node_type=self.node_info.node_type,
            module_name=module_name,
            module_params=module_hyperparams,
            metric_value=target_metric,
            metric_name=self.target_metric,
            metrics=quality_metrics,
            artifact=module.get_assets(),  # retriever name / scores / predictions
            module_dump_dir=self.get_module_dump_dir(context, module_name, self._counter),
            module=module,
        )
        context.dump_optimization_info()

        if context.is_ram_to_clear():
            module.clear_cache()
            gc.collect()
            torch.cuda.empty_cache()

        self._counter += 1
        return target_metric

    def _suggest_module_and_hyperparams(
        self, trial: Trial, search_space: list[dict[str, Any]]
    ) -> tuple[str, dict[str, Any]]:
        """Sample module name and its hyperparams from given search space."""
        n_modules = len(search_space)
        id_module_chosen = trial.suggest_categorical("module_idx", list(range(n_modules)))
        module_chosen = deepcopy(search_space[id_module_chosen])
        module_name = module_chosen.pop("module_name")
        module_config = self._suggest_hyperparams(trial, f"{module_name}_{id_module_chosen}", module_chosen)
        return module_name, module_config

    def _suggest_hyperparams(
        self, trial: Trial, module_name: str, search_space: dict[str, Any | list[Any]]
    ) -> dict[str, Any]:
        res: dict[str, Any] = {}

        for param_name, param_space in search_space.items():
            name = f"{module_name}_{param_name}"
            if isinstance(param_space, list):
                res[param_name] = trial.suggest_categorical(name, choices=param_space)
            elif self._parse_param_space(param_space, ParamSpaceInt):
                res[param_name] = trial.suggest_int(name, **param_space)
            elif self._parse_param_space(param_space, ParamSpaceFloat):
                res[param_name] = trial.suggest_float(name, **param_space)
            else:
                msg = f"Unsupported type of param search space {name}: {param_space}"
                raise TypeError(msg)
        return res

    def _parse_param_space(self, param_space: dict[str, Any], space_type: type[ParamSpaceT]) -> ParamSpaceT | None:
        try:
            return space_type(**param_space)
        except ValueError:
            return None

    def get_module_dump_dir(self, context: Context, module_name: str, j_combination: int) -> str | None:
        """Creates and returns the path to the module dump directory.

        Args:
            context: The context object.
            module_name: The name of the module being optimized.
            j_combination: The combination index for the parameters.

        Returns:
            The path to the module dump directory.
        """
        dump_dir = context.get_dump_dir()
        if dump_dir is None:
            return None
        dump_dir_ = dump_dir / self.node_info.node_type / module_name / f"comb_{j_combination}"
        dump_dir_.mkdir(parents=True, exist_ok=True)
        return str(dump_dir_)

    def validate_nodes_with_dataset(self, dataset: Dataset, mode: SearchSpaceValidationMode) -> None:  # noqa: C901
        """Validates nodes against the dataset.

        Args:
            dataset: The dataset used for validation.
            mode: The validation mode ("raise" or "warning").

        Raises:
            ValueError: If validation fails and `mode` is set to "raise".
        """
        is_multilabel = dataset.multilabel

        filtered_search_space = []
        if is_multilabel and self.target_metric not in self.node_info.multilabel_available_metrics:
            handle_message_on_mode(
                mode,
                f"Target metric '{self.target_metric}' is not available for multilabel datasets. "
                f"Available metrics: {list(self.node_info.multilabel_available_metrics.keys())}",
                True,
            )
        elif not is_multilabel and self.target_metric not in self.node_info.multiclass_available_metrics:
            handle_message_on_mode(
                mode,
                f"Target metric '{self.target_metric}' is not available for multiclass datasets. "
                f"Available metrics: {list(self.node_info.multiclass_available_metrics.keys())}",
                True,
            )

        for metric in self.metrics:
            if is_multilabel and metric not in self.node_info.multilabel_available_metrics:
                handle_message_on_mode(
                    mode,
                    f"Metric '{metric}' is not available for multilabel datasets. "
                    f"Available metrics: {list(self.node_info.multilabel_available_metrics.keys())}",
                    True,
                )
            elif not is_multilabel and metric not in self.node_info.multiclass_available_metrics:
                handle_message_on_mode(
                    mode,
                    f"Metric '{metric}' is not available for multiclass datasets. "
                    f"Available metrics: {list(self.node_info.multiclass_available_metrics.keys())}",
                    True,
                )

        for search_space in deepcopy(self.modules_search_spaces):
            module_name = search_space["module_name"]
            module = self.node_info.modules_available[module_name]
            messages = []

            if module_name == "description" and not dataset.has_descriptions:
                messages.append("DescriptionScorer cannot be used without intents descriptions.")

            if is_multilabel and not module.supports_multilabel:
                messages.append(f"Module '{module_name}' does not support multilabel datasets.")

            if not is_multilabel and not module.supports_multiclass:
                messages.append(f"Module '{module_name}' does not support multiclass datasets.")

            if len(messages) > 0:
                msg = "\n".join(messages)
                handle_message_on_mode(mode, msg)
            else:
                filtered_search_space.append(search_space)

        self.modules_search_spaces = filtered_search_space

    def validate_search_space(self, search_space: list[dict[str, Any]]) -> None:
        """Check if search space is configured correctly."""
        validated_search_space = SearchSpaceConfig(search_space).model_dump()

        if not bool(int(os.getenv("AUTOINTENT_EXTRA_VALIDATION", "0"))):
            return

        for module_search_space in validated_search_space:
            module_search_space_no_optuna, module_name = self._reformat_search_space(deepcopy(module_search_space))

            for params_combination in it.product(*module_search_space_no_optuna.values()):
                module_kwargs = dict(zip(module_search_space_no_optuna.keys(), params_combination, strict=False))

                self._logger.debug("validating %s module...", module_name, extra=module_kwargs)
                module = self.node_info.modules_available[module_name](**module_kwargs)
                self._logger.debug("%s is ok", module_name)

                del module
                gc.collect()

    def _reformat_search_space(self, module_search_space: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Remove optuna notation from search space."""
        res = {}
        module_name = module_search_space.pop("module_name")

        for param_name, param_space in module_search_space.items():
            if param_name == "n_trials":
                continue
            if isinstance(param_space, list):
                res[param_name] = param_space
            elif self._parse_param_space(param_space, ParamSpaceInt) or self._parse_param_space(
                param_space, ParamSpaceFloat
            ):
                res[param_name] = [param_space["low"], param_space["high"]]
            else:
                msg = f"Unsupported type of param search space: {param_space}"
                raise TypeError(msg)

        return res, module_name


def get_storage_url(study_name: str, storage_dir: Path | None) -> str | None:
    """Create SQLite database URL for Optuna study persistence.

    Args:
        study_name: Name of the study to be used as filename
        storage_dir: Directory to store the database file

    Returns:
        SQLite URL for Optuna storage
    """
    if storage_dir is None:
        msg = "Storage directory must be provided for study persistence."
        logger.warning(msg)
        return None
    storage_dir = storage_dir / "optuna_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_path = storage_dir / f"{study_name}.db"
    return f"sqlite:///{db_path}"


def load_or_create_study(
    study_name: str,
    context: Context,
    sampler: optuna.samplers.BaseSampler,
    n_trials: int,
    direction: str = "maximize",
) -> tuple[optuna.Study, int, int]:
    """Load an existing study or create a new one if it doesn't exist.

    Args:
        study_name: Name of the study
        context: Context object
        direction: Optimization direction (maximize or minimize)
        sampler: Optuna sampler instance
        n_trials: n_trials

    Returns:
        Optuna study instance, number of completed trials, and number trials to run
    """
    remaining_trials = n_trials
    finished_trials = 0

    storage_url = get_storage_url(study_name, context.get_dump_dir())

    try:
        # will catch exception if study does not exist
        study = optuna.load_study(study_name=study_name, storage=storage_url, sampler=sampler)  # type: ignore[arg-type]

        if study.trials:
            logger.info(
                "Resuming optimization from previous run. %d trials already completed.",
                len(study.trials),
            )
            # Find the highest trial number to continue counting
            finished_trials = max(t.number for t in study.trials) + 1
            # Calculate remaining trials if n_trials is specified
            remaining_trials = max(0, n_trials - len(study.trials))

        context.load_optimization_info()
        return study, finished_trials, remaining_trials  # noqa: TRY300
    except Exception:  # noqa: BLE001
        # Create a new study if none exists
        return (
            optuna.create_study(  # TODO add pruner?
                study_name=study_name,
                storage=storage_url,
                direction=direction,
                sampler=sampler,
                load_if_exists=True,
            ),
            finished_trials,
            remaining_trials,
        )


def handle_message_on_mode(
    mode: SearchSpaceValidationMode,
    message: str,
    strict: bool = False,
) -> None:
    """Handle messages based on the validation mode.

    Args:
        mode: The validation mode ("raise" or "warning").
        message: The message to handle.
        strict: If True always raises an error, even if mode is "warning".

    Raises:
        ValueError: If mode is "raise".
    """
    if mode == "raise":
        raise ValueError(message)
    if mode == "warning":
        logger.warning(message)
    if strict:
        raise ValueError(message)


# TODO research on possibility to use custom pruner
def _check_duplicate(trial: Trial) -> float | None:
    completed_trials = trial.study.get_trials(states=[optuna.trial.TrialState.COMPLETE], deepcopy=False)

    previous_trial = next(
        (completed_trial for completed_trial in completed_trials if completed_trial.params == trial.params), None
    )

    return previous_trial.value if previous_trial is not None else None
