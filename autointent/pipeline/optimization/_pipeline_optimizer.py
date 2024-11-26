"""Pipeline optimizer."""

import json
import logging
from typing import Any

import numpy as np
from hydra.utils import instantiate
from typing_extensions import Self

from autointent import Context
from autointent.configs._optimization_cli import EmbedderConfig, LoggingConfig, VectorIndexConfig
from autointent.configs._pipeline_optimizer import PipelineOptimizerConfig
from autointent.context.data_handler import Dataset
from autointent.custom_types import NodeType
from autointent.nodes import NodeOptimizer


class PipelineOptimizer:
    """Pipeline optimizer class."""

    def __init__(
        self,
        nodes: list[NodeOptimizer],
    ) -> None:
        """
        Initialize the pipeline optimizer.

        :param nodes: list of nodes
        """
        self._logger = logging.getLogger(__name__)
        self.nodes = nodes

        self.logging_config = LoggingConfig(dump_dir=None)
        self.vector_index_config = VectorIndexConfig()
        self.embedder_config = EmbedderConfig()

    def set_config(self, config: LoggingConfig | VectorIndexConfig | EmbedderConfig) -> None:
        """
        Set configuration for the optimizer.

        :param config: Configuration
        """
        if isinstance(config, LoggingConfig):
            self.logging_config = config
        elif isinstance(config, VectorIndexConfig):
            self.vector_index_config = config
        elif isinstance(config, EmbedderConfig):
            self.embedder_config = config
        else:
            msg = "unknown config type"
            raise TypeError(msg)

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> Self:
        """
        Create pipeline optimizer from dictionary config.

        :param config: Dictionary config
        """
        return instantiate(PipelineOptimizerConfig, **config)  # type: ignore[no-any-return]

    def optimize(self, context: Context) -> None:
        """
        Optimize the pipeline.

        :param context: Context
        """
        self.context = context
        self._logger.info("starting pipeline optimization...")
        for node_optimizer in self.nodes:
            node_optimizer.fit(context)
        if not context.vector_index_config.save_db:
            self._logger.info("removing vector database from file system...")
            context.vector_index_client.delete_db()

    def optimize_from_dataset(self, dataset: Dataset, force_multilabel: bool = False) -> Context:
        """
        Optimize the pipeline from dataset.

        :param dataset: Dataset for optimization
        :param force_multilabel: Whether to force multilabel or not
        :return: Context
        """
        context = Context()
        context.set_dataset(dataset, force_multilabel)
        context.configure_logging(self.logging_config)
        context.configure_vector_index(self.vector_index_config, self.embedder_config)

        self.optimize(context)
        self.inference_config = context.optimization_info.get_inference_nodes_config()
        return context


def make_report(logs: dict[str, Any], nodes: list[NodeType]) -> str:
    """
    Generate a report from optimization logs.

    :param logs: Logs
    :param nodes: Nodes
    :return: String report
    """
    ids = [np.argmax(logs["metrics"][node]) for node in nodes]
    configs = []
    for i, node in zip(ids, nodes, strict=False):
        cur_config = logs["configs"][node][i]
        cur_config["metric_value"] = logs["metrics"][node][i]
        configs.append(cur_config)
    messages = [json.dumps(c, indent=4) for c in configs]
    msg = "\n".join(messages)
    return "resulting pipeline configuration is the following:\n" + msg
