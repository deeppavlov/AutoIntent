import gc
import itertools as it
from copy import deepcopy
from typing import Callable

import torch

from ..context import Context
from ..modules import Module


class Node:
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
                metric, assets = module.fit_score(
                    context, self.metrics_available[self.metric_name]
                )
                context.optimization_logs.log_module_optimization(
                    self.node_type,
                    module_type,
                    module_config,
                    metric,
                    self.metric_name,
                    assets,  # retriever name / scores / predictions
                    self.verbose
                )
                module.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()
