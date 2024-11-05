import json
import logging
from typing import Any

import numpy as np
from hydra.utils import instantiate

from autointent import Context
from autointent.configs.optimization_cli import EmbedderConfig, LoggingConfig, VectorIndexConfig
from autointent.configs.pipeline_optimizer import PipelineOptimizerConfig
from autointent.context.data_handler import Dataset
from autointent.nodes import NodeOptimizer


class PipelineOptimizer:
    def __init__(self, nodes: list[NodeOptimizer]) -> None:
        self._logger = logging.getLogger(__name__)
        self.nodes = nodes

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> "PipelineOptimizer":
        return instantiate(PipelineOptimizerConfig, **config)  # type: ignore[no-any-return]

    def optimize(self, context: Context) -> None:
        self.context = context
        self._logger.info("starting pipeline optimization...")
        for node_optimizer in self.nodes:
            node_optimizer.fit(context)

    def optimize_from_dataset(self, train_data: Dataset, val_data: Dataset | None = None) -> Context:
        context = Context()
        context.set_datasets(train_data, val_data)
        context.config_logs(LoggingConfig(dump_dir=None))
        context.config_vector_index(VectorIndexConfig(), EmbedderConfig())

        self.optimize(context)
        self.inference_config = context.optimization_info.get_inference_nodes_config()
        return context


def make_report(logs: dict[str, Any], nodes: list[str]) -> str:
    ids = [np.argmax(logs["metrics"][node]) for node in nodes]
    configs = []
    for i, node in zip(ids, nodes, strict=False):
        cur_config = logs["configs"][node][i]
        cur_config["metric_value"] = logs["metrics"][node][i]
        configs.append(cur_config)
    messages = [json.dumps(c, indent=4) for c in configs]
    msg = "\n".join(messages)
    return "resulting pipeline configuration is the following:\n" + msg
