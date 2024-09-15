import gc
import itertools as it
from copy import deepcopy
from typing import Callable

import torch

from ..context import Context
from ..modules import Module


class Node:
    @classmethod
    def get_module_key(cls, module_class_name):
        for key, value in cls.modules_available.items():
            if value.__name__ == module_class_name:
                return key
        raise ValueError(f"Module class {module_class_name} not found in modules_available")

    metrics_available: dict[str, Callable]  # metrics functions
    modules_available: dict[str, Callable]  # modules constructors
    node_type: str

    def __init__(self, modules_search_spaces: list[dict], metric: str, verbose: bool = False):
        """
        `modules_search_spaces`: list of records, where each record is a mapping: hyperparam_name -> list of values (search space) with extra field "module_type" with values from ["knn", "linear", "dnnc"]
        """
        self.modules_search_spaces = modules_search_spaces
        self.metric_name = metric
        self.verbose = verbose

    def fit(self, context: Context):
        for search_space in deepcopy(self.modules_search_spaces):
            module_type = search_space.pop("module_type")
            for module_config in it.product(*search_space.values()):
                module_config = dict(zip(search_space.keys(), module_config))
                module: Module = self.modules_available[module_type](**module_config)
                module.fit(context)
                metric_value = module.score(context, self.metrics_available[self.metric_name])
                assets = module.get_assets(context)
                context.optimization_logs.log_module_optimization(
                    self.node_type,
                    module_type,
                    module_config,
                    metric_value,
                    self.metric_name,
                    assets,  # retriever name / scores / predictions
                    self.verbose
                )
                module.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()
