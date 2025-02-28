"""Node optimizer for optimizing module configurations."""

import gc
import itertools as it
import logging
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.trial import Trial
from pydantic import BaseModel, Field
from typing_extensions import assert_never

from autointent import Dataset
from autointent.context import Context
from autointent.custom_types import NodeType, SamplerType, SearchSpaceValidationMode
from autointent.nodes.info import NODES_INFO


class ParamSpaceInt(BaseModel):
    """Integer parameter search space configuration."""

    low: int = Field(..., description="Lower boundary of the search space.")
    high: int = Field(..., description="Upper boundary of the search space.")
    step: int = Field(1, description="Step size for the search space.")
    log: bool = Field(False, description="Indicates whether to use a logarithmic scale.")


class ParamSpaceFloat(BaseModel):
    """Float parameter search space configuration."""

    low: float = Field(..., description="Lower boundary of the search space.")
    high: float = Field(..., description="Upper boundary of the search space.")
    step: float | None = Field(None, description="Step size for the search space (if applicable).")
    log: bool = Field(False, description="Indicates whether to use a logarithmic scale.")


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

    def fit(self, context: Context, sampler: SamplerType = "brute") -> None:
        """Performs the optimization process for the node.

        Args:
            context: The optimization context containing relevant data.
            sampler: The sampling strategy used for optimization.

        Raises:
            AssertionError: If an invalid sampler type is provided.
        """
        self._logger.info("Starting %s node optimization...", self.node_info.node_type)

        for search_space in deepcopy(self.modules_search_spaces):
            self._counter: int = 0
            module_name = search_space.pop("module_name")
            n_trials = search_space.pop("n_trials", None)

            if sampler == "tpe":
                sampler_instance = optuna.samplers.TPESampler(seed=context.seed)
                n_trials = n_trials or 10
            elif sampler == "brute":
                sampler_instance = optuna.samplers.BruteForceSampler(seed=context.seed)  # type: ignore[assignment]
                n_trials = None
            elif sampler == "random":
                sampler_instance = optuna.samplers.RandomSampler(seed=context.seed)  # type: ignore[assignment]
                n_trials = n_trials or 10
            else:
                assert_never(sampler)

            study = optuna.create_study(direction="maximize", sampler=sampler_instance)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            obj = partial(self.objective, module_name=module_name, search_space=search_space, context=context)
            study.optimize(obj, n_trials=n_trials)

        self._logger.info("%s node optimization is finished!", self.node_info.node_type)

    def objective(
        self,
        trial: Trial,
        module_name: str,
        search_space: dict[str, ParamSpaceInt | ParamSpaceFloat | list[Any]],
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
        config = self.suggest(trial, search_space)

        self._logger.debug("Initializing %s module...", module_name)
        module = self.node_info.modules_available[module_name].from_context(context, **config)

        embedder_config = module.get_embedder_config()
        if embedder_config is not None:
            config["embedder_config"] = embedder_config

        context.callback_handler.start_module(module_name=module_name, num=self._counter, module_kwargs=config)

        self._logger.debug("Scoring %s module...", module_name)
        all_metrics = module.score(context, metrics=self.metrics)
        target_metric = all_metrics[self.target_metric]

        context.callback_handler.log_metrics(all_metrics)
        context.callback_handler.end_module()

        dump_dir = context.get_dump_dir()

        if dump_dir is not None:
            module_dump_dir = self.get_module_dump_dir(dump_dir, module_name, self._counter)
            module.dump(module_dump_dir)
        else:
            module_dump_dir = None

        context.optimization_info.log_module_optimization(
            self.node_info.node_type,
            module_name,
            config,
            target_metric,
            self.target_metric,
            all_metrics,
            module.get_assets(),  # retriever name / scores / predictions
            module_dump_dir,
            module=module if not context.is_ram_to_clear() else None,
        )

        if context.is_ram_to_clear():
            module.clear_cache()
            gc.collect()
            torch.cuda.empty_cache()

        self._counter += 1
        return target_metric

    def suggest(self, trial: Trial, search_space: dict[str, Any | list[Any]]) -> dict[str, Any]:
        """Suggests parameter values based on the search space.

        Args:
            trial: The Optuna trial instance.
            search_space: A dictionary defining the parameter search space.

        Returns:
            A dictionary containing the suggested parameter values.

        Raises:
            TypeError: If an unsupported parameter search space type is encountered.
        """
        res: dict[str, Any] = {}

        for param_name, param_space in search_space.items():
            if isinstance(param_space, list):
                res[param_name] = trial.suggest_categorical(param_name, choices=param_space)
            elif self._is_valid_param_space(param_space, ParamSpaceInt):
                res[param_name] = trial.suggest_int(param_name, **param_space)
            elif self._is_valid_param_space(param_space, ParamSpaceFloat):
                res[param_name] = trial.suggest_float(param_name, **param_space)
            else:
                msg = f"Unsupported type of param search space: {param_space}"
                raise TypeError(msg)
        return res

    def _is_valid_param_space(
        self, param_space: dict[str, Any], space_type: type[ParamSpaceInt | ParamSpaceFloat]
    ) -> bool:
        try:
            space_type(**param_space)
            return True  # noqa: TRY300
        except ValueError:
            return False

    def get_module_dump_dir(self, dump_dir: Path, module_name: str, j_combination: int) -> str:
        """Creates and returns the path to the module dump directory.

        Args:
            dump_dir: The base directory for storing module dumps.
            module_name: The name of the module being optimized.
            j_combination: The combination index for the parameters.

        Returns:
            The path to the module dump directory.
        """
        dump_dir_ = dump_dir / self.node_info.node_type / module_name / f"comb_{j_combination}"
        dump_dir_.mkdir(parents=True, exist_ok=True)
        return str(dump_dir_)

    def validate_nodes_with_dataset(self, dataset: Dataset, mode: SearchSpaceValidationMode) -> None:
        """Validates nodes against the dataset.

        Args:
            dataset: The dataset used for validation.
            mode: The validation mode ("raise" or "warning").

        Raises:
            ValueError: If validation fails and `mode` is set to "raise".
        """
        is_multilabel = dataset.multilabel

        filtered_search_space = []

        for search_space in deepcopy(self.modules_search_spaces):
            module_name = search_space["module_name"]
            module = self.node_info.modules_available[module_name]
            # todo add check for oos

            messages = []

            if module_name == "description" and not dataset.has_descriptions:
                messages.append("DescriptionScorer cannot be used without intents descriptions.")

            if is_multilabel and not module.supports_multilabel:
                messages.append(f"Module '{module_name}' does not support multilabel datasets.")

            if not is_multilabel and not module.supports_multiclass:
                messages.append(f"Module '{module_name}' does not support multiclass datasets.")

            if len(messages) > 0:
                msg = "\n".join(messages)
                if mode == "raise":
                    self._logger.error(msg)
                    raise ValueError(msg)
                if mode == "warning":
                    self._logger.warning(msg)
            else:
                filtered_search_space.append(search_space)

        self.modules_search_spaces = filtered_search_space

    def validate_search_space(self, search_space: list[dict[str, Any]]) -> None:
        """Check if search space is configured correctly."""
        for module_search_space in search_space:
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
            elif self._is_valid_param_space(param_space, ParamSpaceInt) or self._is_valid_param_space(
                param_space, ParamSpaceFloat
            ):
                res[param_name] = [param_space["low"], param_space["high"]]
            else:
                msg = f"Unsupported type of param search space: {param_space}"
                raise TypeError(msg)

        return res, module_name
