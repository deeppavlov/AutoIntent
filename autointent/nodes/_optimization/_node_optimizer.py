"""Node optimizer."""

import gc
import logging
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.trial import Trial
from pydantic import BaseModel, Field

from autointent import Dataset
from autointent.context import Context
from autointent.custom_types import NodeType, SamplerType
from autointent.nodes._nodes_info import NODES_INFO


class ParamSpaceInt(BaseModel):
    low: int = Field(..., description="Low boundary of the search space.")
    high: int = Field(..., description="High boundary of the search space.")
    step: int = Field(1, description="Step of the search space.")
    log: bool = Field(False, description="Whether to use a logarithmic scale.")


class ParamSpaceFloat(BaseModel):
    low: float = Field(..., description="Low boundary of the search space.")
    high: float = Field(..., description="High boundary of the search space.")
    step: float | None = Field(None, description="Step of the search space.")
    log: bool = Field(False, description="Whether to use a logarithmic scale.")


class NodeOptimizer:
    """Node optimizer class."""

    def __init__(
        self,
        node_type: NodeType,
        search_space: list[dict[str, Any]],
        target_metric: str,
        metrics: list[str] | None = None,
    ) -> None:
        """
        Initialize the node optimizer.

        :param node_type: Node type
        :param search_space: Search space for the optimization
        :param metrics: Metrics to optimize.
        """
        self.node_type = node_type
        self.node_info = NODES_INFO[node_type]
        self.target_metric = target_metric

        self.metrics = metrics if metrics is not None else []
        if self.target_metric not in self.metrics:
            self.metrics.append(self.target_metric)

        self.modules_search_spaces = search_space
        self._logger = logging.getLogger(__name__)  # TODO solve duplicate logging messages problem

    def fit(self, context: Context, sampler: SamplerType = "brute") -> None:
        """
        Fit the node optimizer.

        :param context: Context
        """
        self._logger.info("starting %s node optimization...", self.node_info.node_type)

        for search_space in deepcopy(self.modules_search_spaces):
            self._counter = 0
            module_name = search_space.pop("module_name")
            n_trials = None
            if "n_trials" in search_space:
                n_trials = search_space.pop("n_trials")
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
                msg = f"Unexpected sampler: {sampler}"
                raise ValueError(msg)
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
        config = self.suggest(trial, search_space)

        self._logger.debug("initializing %s module...", module_name)
        module = self.node_info.modules_available[module_name].from_context(context, **config)

        embedder_config = module.get_embedder_config()
        if embedder_config is not None:
            config["embedder_config"] = embedder_config

        context.callback_handler.start_module(module_name=module_name, num=self._counter, module_kwargs=config)

        self._logger.debug("scoring %s module...", module_name)
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
        res: dict[str, Any] = {}

        def is_valid_param_space(
            param_space: dict[str, Any], space_type: type[ParamSpaceInt | ParamSpaceFloat]
        ) -> bool:
            try:
                space_type(**param_space)
                return True  # noqa: TRY300
            except ValueError:
                return False

        for param_name, param_space in search_space.items():
            if isinstance(param_space, list):
                res[param_name] = trial.suggest_categorical(param_name, choices=param_space)
            elif is_valid_param_space(param_space, ParamSpaceInt):
                res[param_name] = trial.suggest_int(param_name, **param_space)
            elif is_valid_param_space(param_space, ParamSpaceFloat):
                res[param_name] = trial.suggest_float(param_name, **param_space)
            else:
                msg = f"Unsupported type of param search space: {param_space}"
                raise TypeError(msg)
        return res

    def get_module_dump_dir(self, dump_dir: Path, module_name: str, j_combination: int) -> str:
        """
        Get module dump directory.

        :param dump_dir: The base directory where the module dump directories will be created.
        :param module_name: The type of the module being optimized.
        :param j_combination: The index of the parameter combination being used.
        :return: The path to the module dump directory as a string.
        """
        dump_dir_ = dump_dir / self.node_info.node_type / module_name / f"comb_{j_combination}"
        dump_dir_.mkdir(parents=True, exist_ok=True)
        return str(dump_dir_)

    def validate_nodes_with_dataset(self, dataset: Dataset) -> None:
        """
        Validate nodes with dataset.

        :param dataset: Dataset to use
        """
        is_multilabel = dataset.multilabel

        for search_space in deepcopy(self.modules_search_spaces):
            module_name = search_space.pop("module_name")
            module = self.node_info.modules_available[module_name]
            # todo add check for oos

            if is_multilabel and not module.supports_multilabel:
                msg = f"Module '{module_name}' does not support multilabel datasets."
                self._logger.error(msg)
                raise ValueError(msg)
            if not is_multilabel and not module.supports_multiclass:
                msg = f"Module '{module_name}' does not support multiclass datasets."
                self._logger.error(msg)
                raise ValueError(msg)
