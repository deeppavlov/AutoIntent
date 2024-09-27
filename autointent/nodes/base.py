import gc
import itertools as it
from collections.abc import Callable
from copy import deepcopy
from logging import Logger
from typing import TYPE_CHECKING

import torch

from autointent.context import Context

if TYPE_CHECKING:
    from autointent.modules import Module


class Node:
    metrics_available: dict[str, Callable]  # metrics functions
    modules_available: dict[str, Callable]  # modules constructors
    node_type: str

    def __init__(self, modules_search_spaces: list[dict], metric: str, logger: Logger):
        """
        `modules_search_spaces`: list of records, where each record is a mapping: hyperparam_name -> list of values (search space) with extra field "module_type" with values from ["knn", "linear", "dnnc"]
        """
        self._logger = logger
        self.modules_search_spaces = modules_search_spaces
        self.metric_name = metric

    def fit(self, context: Context):
        self._logger.info(f"starting {self.node_type} node optimization...")

        for search_space in deepcopy(self.modules_search_spaces):
            module_type = search_space.pop("module_type")
            for module_config in it.product(*search_space.values()):
                module_config = dict(zip(search_space.keys(), module_config, strict=False))

                self._logger.debug(f"initializing {module_type} module...")
                module: Module = self.modules_available[module_type](**module_config)

                self._logger.debug(f"optimizing {module_type} module...")
                module.fit(context)

                self._logger.debug(f"scoring {module_type} module...")
                metric_value = module.score(context, self.metrics_available[self.metric_name])

                assets = module.get_assets()
                context.optimization_logs.log_module_optimization(
                    self.node_type,
                    module_type,
                    module_config,
                    metric_value,
                    self.metric_name,
                    assets,  # retriever name / scores / predictions
                )
                module.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()
        self._logger.info(f"{self.node_type} node optimization is finished!")
