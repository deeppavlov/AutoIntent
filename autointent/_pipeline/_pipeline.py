"""Pipeline optimizer."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from autointent import Context, Dataset
from autointent.configs import EmbedderConfig, InferenceNodeConfig, LoggingConfig, VectorIndexConfig
from autointent.custom_types import NodeType
from autointent.nodes import InferenceNode, NodeOptimizer
from autointent.utils import load_default_search_space, load_search_space

from ._schemas import InferencePipelineOutput, InferencePipelineUtteranceOutput


class Pipeline:
    """Pipeline optimizer class."""

    def __init__(
        self,
        nodes: list[NodeOptimizer] | list[InferenceNode],
    ) -> None:
        """
        Initialize the pipeline optimizer.

        :param nodes: list of nodes
        """
        self._logger = logging.getLogger(__name__)
        self.nodes = {node.node_type: node for node in nodes}

        if isinstance(nodes[0], NodeOptimizer):
            self.logging_config = LoggingConfig(dump_dir=None)
            self.vector_index_config = VectorIndexConfig()
            self.embedder_config = EmbedderConfig()
        elif not isinstance(nodes[0], InferenceNode):
            msg = "Pipeline should be initialized with list of NodeOptimizers or InferenceNodes"
            raise TypeError(msg)

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
    def from_search_space(cls, search_space: list[dict[str, Any]] | Path | str) -> "Pipeline":
        """
        Create pipeline optimizer from dictionary search space.

        :param config: Dictionary config
        """
        if isinstance(search_space, Path | str):
            search_space = load_search_space(search_space)
        if isinstance(search_space, list):
            nodes = [NodeOptimizer(**node) for node in search_space]
        return cls(nodes)

    @classmethod
    def default_optimizer(cls, multilabel: bool) -> "Pipeline":
        """
        Create pipeline optimizer with default search space for given classification task.

        :param multilabel: Wether the task multi-label, or single-label.
        """
        return cls.from_search_space(load_default_search_space(multilabel))

    def _fit(self, context: Context) -> None:
        """
        Optimize the pipeline.

        :param context: Context
        """
        self.context = context
        self._logger.info("starting pipeline optimization...")
        for node_type in NodeType:
            node_optimizer = self.nodes.get(node_type, None)
            if node_optimizer is not None:
                node_optimizer.fit(context)  # type: ignore[union-attr]
        if not context.vector_index_config.save_db:
            self._logger.info("removing vector database from file system...")
            context.vector_index_client.delete_db()

    def _is_inference(self) -> bool:
        """
        Check the mode in which pipeline is.

        :return: True if pipeline is in inference mode, False if in optimization mode.
        """
        return isinstance(self.nodes[NodeType.scoring], InferenceNode)

    def fit(self, dataset: Dataset, force_multilabel: bool = False, init_for_inference: bool = True) -> Context:
        """
        Optimize the pipeline from dataset.

        :param dataset: Dataset for optimization
        :param force_multilabel: Whether to force multilabel or not
        :return: Context
        """
        if self._is_inference():
            msg = "Pipeline in inference mode cannot be fitted"
            raise RuntimeError(msg)

        context = Context()
        context.set_dataset(dataset, force_multilabel)
        context.configure_logging(self.logging_config)
        context.configure_vector_index(self.vector_index_config, self.embedder_config)

        self._fit(context)

        if init_for_inference:
            if context.is_ram_to_clear():
                nodes_configs = context.optimization_info.get_inference_nodes_config()
                nodes_list = [InferenceNode.from_config(cfg) for cfg in nodes_configs]
            else:
                modules_dict = context.optimization_info.get_best_modules()
                nodes_list = [InferenceNode(module, node_type) for node_type, module in modules_dict.items()]

            self.nodes = {node.node_type: node for node in nodes_list}

        return context

    @classmethod
    def from_dict_config(cls, nodes_configs: list[dict[str, Any]]) -> "Pipeline":
        """
        Create inference pipeline from dictionary config.

        :param nodes_configs: list of dictionary config for nodes
        :return: pipeline ready for inference
        """
        return cls.from_config([InferenceNodeConfig(**cfg) for cfg in nodes_configs])

    @classmethod
    def from_config(cls, nodes_configs: list[InferenceNodeConfig]) -> "Pipeline":
        """
        Create inference pipeline from config.

        :param nodes_configs: list of config for nodes
        """
        nodes = [InferenceNode.from_config(cfg) for cfg in nodes_configs]
        return cls(nodes)

    @classmethod
    def load(cls, path: str | Path) -> "Pipeline":
        """
        Load pipeline in inference mode.

        This method loads fitted modules and tuned hyperparameters.
        :path: path to optimization run directory
        :return: initialized pipeline, ready for inference
        """
        with (Path(path) / "inference_config.yaml").open() as file:
            inference_dict_config = yaml.safe_load(file)
        return cls.from_dict_config(inference_dict_config["nodes_configs"])

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """
        Predict the labels for the utterances.

        :param utterances: list of utterances
        :return: list of predicted labels
        """
        if not self._is_inference():
            msg = "Pipeline in optimization mode cannot perform inference"
            raise RuntimeError(msg)

        scores = self.nodes[NodeType.scoring].module.predict(utterances)  # type: ignore[union-attr]
        return self.nodes[NodeType.prediction].module.predict(scores)  # type: ignore[union-attr]

    def predict_with_metadata(self, utterances: list[str]) -> InferencePipelineOutput:
        """
        Predict the labels for the utterances with metadata.

        :param utterances: list of utterances
        :return: prediction output
        """
        if not self._is_inference():
            msg = "Pipeline in optimization mode cannot perform inference"
            raise RuntimeError(msg)

        scores, scores_metadata = self.nodes[NodeType.scoring].module.predict_with_metadata(utterances)  # type: ignore[union-attr]
        predictions = self.nodes[NodeType.prediction].module.predict(scores)  # type: ignore[union-attr]
        regexp_predictions, regexp_predictions_metadata = None, None
        if NodeType.regexp in self.nodes:
            regexp_predictions, regexp_predictions_metadata = self.nodes[NodeType.regexp].module.predict_with_metadata(  # type: ignore[union-attr]
                utterances,
            )

        outputs = []
        for idx, utterance in enumerate(utterances):
            output = InferencePipelineUtteranceOutput(
                utterance=utterance,
                prediction=predictions[idx],
                regexp_prediction=regexp_predictions[idx] if regexp_predictions is not None else None,
                regexp_prediction_metadata=regexp_predictions_metadata[idx]
                if regexp_predictions_metadata is not None
                else None,
                score=scores[idx],
                score_metadata=scores_metadata[idx] if scores_metadata is not None else None,
            )
            outputs.append(output)

        return InferencePipelineOutput(
            predictions=predictions,
            regexp_predictions=regexp_predictions,
            utterances=outputs,
        )


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
