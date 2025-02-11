"""Node optimizer."""

import gc
import itertools as it
import logging
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypedDict

import optuna
import torch
from optuna.trial import Trial

from autointent.context import Context
from autointent.custom_types import NodeType, TuningType
from autointent.nodes._nodes_info import NODES_INFO


class ParamSpaceCat(TypedDict):
    choices: list[Any]

class ParamSpaceInt(TypedDict, total=False):
    low: int
    high: int
    step: int
    log: bool

class ParamSpaceFloat(TypedDict, total=False):
    low: float
    high: float
    step: float
    log: bool

class ParamSpace(TypedDict):
    type: Literal["cat", "int", "float"]
    content: ParamSpaceCat | ParamSpaceInt | ParamSpaceFloat


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
        :param metric: Metric to optimize.
        """
        self.node_type = node_type
        self.node_info = NODES_INFO[node_type]
        self.target_metric = target_metric

        self.metrics = metrics if metrics is not None else []
        if self.target_metric not in self.metrics:
            self.metrics.append(self.target_metric)

        self.modules_search_spaces = search_space  # TODO search space validation
        self._logger = logging.getLogger(__name__)  # TODO solve duplicate logging messages problem

    def fit(self, context: Context, tuning: TuningType = "brute") -> None:
        """
        Fit the node optimizer.

        :param context: Context
        """
        self._logger.info("starting %s node optimization...", self.node_info.node_type)

        if tuning == "brute":
            self._fit_brute(context)
        elif tuning == "bayes":
            self._fit_bayes(context)
        else:
            msg = f"Unexepected tuning type: {tuning}"
            raise ValueError(msg)

        self._logger.info("%s node optimization is finished!", self.node_info.node_type)

    def _fit_brute(self, context: Context) -> None:
        for search_space in deepcopy(self.modules_search_spaces):
            module_name = search_space.pop("module_name")

            for j_combination, params_combination in enumerate(it.product(*search_space.values())):
                module_kwargs = dict(zip(search_space.keys(), params_combination, strict=False))

                self._logger.debug("initializing %s module...", module_name)
                module = self.node_info.modules_available[module_name].from_context(context, **module_kwargs)

                embedder_name = module.get_embedder_name()
                if embedder_name is not None:
                    module_kwargs["embedder_name"] = embedder_name

                context.callback_handler.start_module(
                    module_name=module_name, num=j_combination, module_kwargs=module_kwargs
                )

                self._logger.debug("scoring %s module...", module_name)
                metrics_score = module.score(context, metrics=self.metrics)
                metric_value = metrics_score[self.target_metric]

                context.callback_handler.log_metrics(metrics_score)
                context.callback_handler.end_module()

                dump_dir = context.get_dump_dir()

                if dump_dir is not None:
                    module_dump_dir = self.get_module_dump_dir(dump_dir, module_name, j_combination)
                    module.dump(module_dump_dir)
                else:
                    module_dump_dir = None

                context.optimization_info.log_module_optimization(
                    self.node_info.node_type,
                    module_name,
                    module_kwargs,
                    metric_value,
                    self.target_metric,
                    module.get_assets(),  # retriever name / scores / predictions
                    module_dump_dir,
                    module=module if not context.is_ram_to_clear() else None,
                )

                if context.is_ram_to_clear():
                    module.clear_cache()
                    gc.collect()
                    torch.cuda.empty_cache()

    def _fit_bayes(self, context: Context, seed: int = 42, n_trials: int = 10) -> None:
        self._counter = 0
        for search_space in deepcopy(self.modules_search_spaces):
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            module_name = search_space.pop("module_name")
            obj = partial(self.objective, module_name=module_name, search_space=search_space, context=context)
            study.optimize(obj, n_trials=n_trials)

    def objective(
        self, trial: Trial, module_name: str, search_space: dict[str, ParamSpace | list[Any]], context: Context
    ) -> float:
        config = self.suggest(trial, search_space)

        self._logger.debug("initializing %s module...", module_name)
        module = self.node_info.modules_available[module_name].from_context(context, **config)

        embedder_name = module.get_embedder_name()
        if embedder_name is not None:
            config["embedder_name"] = embedder_name

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

    def suggest(self, trial: Trial, search_space: dict[str, ParamSpace | list[Any]]) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for param_name, param_space in search_space.items():
            if isinstance(param_space, list):
                res[param_name] = trial.suggest_categorical(param_name, choices=param_space)
            elif param_space["type"] == "cat":
                res[param_name] = trial.suggest_categorical(param_name, **param_space["content"])
            elif param_space["type"] == "int":
                res[param_name] = trial.suggest_int(param_name, **param_space["content"])
            elif param_space["type"] == "float":
                res[param_name] = trial.suggest_float(param_name, **param_space["content"])
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
