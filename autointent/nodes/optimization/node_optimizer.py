import gc
import itertools as it
import logging
from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from hydra.utils import instantiate

from autointent.context import Context
from autointent.nodes.nodes_info import NODES_INFO

if TYPE_CHECKING:
    from autointent.modules import Module


class NodeOptimizer:
    def __init__(self, node_type: str, search_space: list[dict], metric: str):
        self.node_info = NODES_INFO[node_type]
        self.metric_name = metric
        self.modules_search_spaces = search_space  # TODO search space validation
        self._logger = logging.getLogger(__name__)

    def fit(self, context: Context):
        self._logger.info("starting %s node optimization...", self.node_info.node_type)

        for search_space in deepcopy(self.modules_search_spaces):
            module_type = search_space.pop("module_type")

            for params_combination in it.product(*search_space.values()):
                module_kwargs = dict(zip(search_space.keys(), params_combination, strict=False))

                self._logger.debug("initializing %s module...", module_type)
                module_config = self.node_info.modules_configs[module_type](**module_kwargs)
                module: Module = instantiate(module_config)

                self._logger.debug("optimizing %s module...", module_type)
                module.fit(context)

                self._logger.debug("scoring %s module...", module_type)
                metric_value = module.score(context, self.node_info.metrics_available[self.metric_name])

                assets = module.get_assets()
                context.optimization_info.log_module_optimization(
                    self.node_info.node_type,
                    module_type,
                    module_kwargs,
                    metric_value,
                    self.metric_name,
                    assets,  # retriever name / scores / predictions
                )
                module.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()
        self._logger.info("%s node optimization is finished!", self.node_info.node_type)
