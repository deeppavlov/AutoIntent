import gc
import itertools as it
from abc import ABC
from collections.abc import Callable
from copy import deepcopy
from logging import Logger

import torch

from autointent.context import Context
from autointent.modules import Module


class Node(ABC):
    metrics_available: dict[str, Callable]  # metrics functions
    modules_available: dict[str, type[Module]]  # modules constructors
    node_type: str


class OptimizationNode(Node):
    def configure_optimization(self, modules_search_spaces: list[dict], metric: str, logger: Logger):
        """
        `modules_search_spaces`: list of records, where each record is a mapping: hyperparam_name -> list of values \
            (search space) with extra field "module_type" with values from ["knn", "linear", "dnnc"]
        """
        self._logger = logger
        self.modules_search_spaces = modules_search_spaces
        self.metric_name = metric

    def fit(self, context: Context):
        self._logger.info("starting %s node optimization...", self.node_type)

        for search_space in deepcopy(self.modules_search_spaces):
            module_type = search_space.pop("module_type")
            for module_config in it.product(*search_space.values()):
                module_kwargs = dict(zip(search_space.keys(), module_config, strict=False))

                self._logger.debug("initializing %s module...", module_type)
                module = self.modules_available[module_type](**module_kwargs)

                self._logger.debug("optimizing %s module...", module_type)
                module.fit(context)

                self._logger.debug("scoring %s module...", module_type)
                metric_value = module.score(context, self.metrics_available[self.metric_name])

                assets = module.get_assets()
                context.optimization_info.log_module_optimization(
                    self.node_type,
                    module_type,
                    module_kwargs,
                    metric_value,
                    self.metric_name,
                    assets,  # retriever name / scores / predictions
                )
                module.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()
        self._logger.info("%s node optimization is finished!", self.node_type)


class InferenceNode(Node):
    def configure_inference(self):
        ...

    def load(self):
        ...
