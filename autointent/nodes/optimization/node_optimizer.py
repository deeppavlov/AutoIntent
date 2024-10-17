import gc
import itertools as it
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

import torch
from hydra.utils import instantiate

from autointent.configs.node import NodeOptimizerConfig
from autointent.context import Context
from autointent.nodes.nodes_info import NODES_INFO

NodeOptimizerType = TypeVar("NodeOptimizerType", bound="NodeOptimizer")


class NodeOptimizer:
    def __init__(self, node_type: str, search_space: list[dict], metric: str) -> None:
        self.node_info = NODES_INFO[node_type]
        self.metric_name = metric
        self.modules_search_spaces = search_space  # TODO search space validation
        self._logger = logging.getLogger(__name__)  # TODO solve duplicate logging messages problem

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> NodeOptimizerType:
        return instantiate(NodeOptimizerConfig, **config)

    def fit(self, context: Context) -> None:
        self._logger.info("starting %s node optimization...", self.node_info.node_type)

        for search_space in deepcopy(self.modules_search_spaces):
            module_type = search_space.pop("module_type")

            for j_combination, params_combination in enumerate(it.product(*search_space.values())):
                module_kwargs = dict(zip(search_space.keys(), params_combination, strict=False))

                self._logger.debug("initializing %s module...", module_type)
                module = self.node_info.modules_available[module_type](**module_kwargs)

                self._logger.debug("optimizing %s module...", module_type)
                module.fit(context)

                self._logger.debug("scoring %s module...", module_type)
                metric_value = module.score(context, self.node_info.metrics_available[self.metric_name])

                assets = module.get_assets()
                module_dump_dir = self.get_module_dump_dir(context.dump_dir, module_type, j_combination)
                module.dump(module_dump_dir)

                context.optimization_info.log_module_optimization(
                    self.node_info.node_type,
                    module_type,
                    module_kwargs,
                    metric_value,
                    self.metric_name,
                    assets,  # retriever name / scores / predictions
                    module_dump_dir
                )

                module.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()

        self._logger.info("%s node optimization is finished!", self.node_info.node_type)

    def get_module_dump_dir(self, dump_dir: str, module_type: str, j_combination: int) -> str:
        dump_dir_ = Path(dump_dir) / self.node_info.node_type / module_type / f"comb_{j_combination}"
        dump_dir_.mkdir(parents=True, exist_ok=True)
        return str(dump_dir_)
