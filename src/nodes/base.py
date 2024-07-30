import itertools as it
from copy import deepcopy
from typing import Callable

import numpy as np

from ..data_handler import DataHandler
from ..modules import Module


class Node:
    metrics_available: dict[str, Callable]  # metrics functions
    modules_available: dict[str, Callable]  # modules constructors

    def __init__(self, modules_search_spaces: list[dict], metric: str):
        """
        `modules_search_spaces`: list of records, where each record is a mapping: hyperparam_name -> list of values (search space) with extra field "module_type" with values from ["knn", "linear", "dnnc"]
        """
        self.modules_search_spaces = modules_search_spaces
        self.metric_name = metric

    def fit(self, data_handler: DataHandler):
        metric_scores = []
        modules_configs = []
        for search_space in deepcopy(self.modules_search_spaces):
            module_type = search_space.pop("module_type")
            for module_config in it.product(*search_space.values()):
                module_config = dict(zip(search_space.keys(), module_config))
                modules_configs.append(module_config)
                module: Module = self.modules_available[module_type](**module_config)
                metric = module.fit_score(
                    data_handler, self.metrics_available[self.metric_name]
                )
                metric_scores.append(metric)

        self.optimization_results = {
            "metric_name": self.metric_name,
            "i_best": np.argmax(metric_scores),
            "scores": metric_scores,
            "configs": modules_configs,
        }
