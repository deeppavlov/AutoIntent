import gc
import itertools as it
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from typing_extensions import Self

from autointent.configs.node import NodeOptimizerConfig
from autointent.context import Context
from autointent.modules import Module
from autointent.modules.prediction.base import get_prediction_evaluation_data
from autointent.nodes.nodes_info import NODES_INFO


class NodeOptimizer:
    def __init__(self, node_type: str, search_space: list[dict[str, Any]], metric: str) -> None:
        self.node_info = NODES_INFO[node_type]
        self.metric_name = metric
        self.modules_search_spaces = search_space  # TODO search space validation
        self._logger = logging.getLogger(__name__)  # TODO solve duplicate logging messages problem

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> Self:
        return instantiate(NodeOptimizerConfig, **config)  # type: ignore[no-any-return]

    def fit(self, context: Context) -> None:
        self._logger.info("starting %s node optimization...", self.node_info.node_type)

        for search_space in deepcopy(self.modules_search_spaces):
            module_type = search_space.pop("module_type")

            for j_combination, params_combination in enumerate(it.product(*search_space.values())):
                module_kwargs = dict(zip(search_space.keys(), params_combination, strict=False))

                self._logger.debug("initializing %s module...", module_type)
                module = self.node_info.modules_available[module_type].from_context(context, **module_kwargs)

                embedder_name = module.get_embedder_name()
                if embedder_name is not None:
                    module_kwargs["embedder_name"] = embedder_name

                self._logger.debug("optimizing %s module...", module_type)
                self.module_fit(module, context)

                self._logger.debug("scoring %s module...", module_type)
                metric_value = module.score(context, self.node_info.metrics_available[self.metric_name])

                assets = module.get_assets()

                dump_dir = context.get_dump_dir()

                if dump_dir is not None:
                    module_dump_dir = self.get_module_dump_dir(dump_dir, module_type, j_combination)
                    module.dump(module_dump_dir)
                else:
                    module_dump_dir = None

                context.optimization_info.log_module_optimization(
                    self.node_info.node_type,
                    module_type,
                    module_kwargs,
                    metric_value,
                    self.metric_name,
                    assets,  # retriever name / scores / predictions
                    module_dump_dir,
                    module=module if not context.is_ram_to_clear() else None,
                )

                if context.is_ram_to_clear():
                    module.clear_cache()
                    gc.collect()
                    torch.cuda.empty_cache()

        self._logger.info("%s node optimization is finished!", self.node_info.node_type)

    def get_module_dump_dir(self, dump_dir: Path, module_type: str, j_combination: int) -> str:
        dump_dir_ = dump_dir / self.node_info.node_type / module_type / f"comb_{j_combination}"
        dump_dir_.mkdir(parents=True, exist_ok=True)
        return str(dump_dir_)

    def module_fit(self, module: Module, context: Context) -> None:
        if self.node_info.node_type in ["retrieval", "scoring"]:
            if module.__class__.__name__ == "DescriptionScorer":
                args = (
                    context.data_handler.utterances_train,
                    context.data_handler.labels_train,
                    context.data_handler.label_description,
                )
            else:
                args = (context.data_handler.utterances_train, context.data_handler.labels_train)  # type: ignore[assignment]
        elif self.node_info.node_type == "prediction":
            labels, scores = get_prediction_evaluation_data(context)
            args = (scores, labels, context.data_handler.tags)  # type: ignore[assignment]
        else:
            msg = "something's wrong"
            self._logger.error(msg)
            raise ValueError(msg)
        module.fit(*args)  # type: ignore[arg-type]
