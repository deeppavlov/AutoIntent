"""Inference pipeline for prediction."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from typing_extensions import Self

from autointent.configs import InferenceNodeConfig
from autointent.context import Context
from autointent.custom_types import LabelType, NodeType
from autointent.nodes.inference import InferenceNode


class InferencePipelineUtteranceOutput(BaseModel):
    """Output of the inference pipeline for a single utterance."""

    utterance: str
    prediction: LabelType
    regexp_prediction: LabelType | None
    regexp_prediction_metadata: Any
    score: list[float]
    score_metadata: Any


class InferencePipelineOutput(BaseModel):
    """Output of the inference pipeline."""

    predictions: list[LabelType]
    regexp_predictions: list[LabelType] | None = None
    utterances: list[InferencePipelineUtteranceOutput] | None = None


class InferencePipeline:
    """Pipeline for inference."""

    def __init__(self, nodes: list[InferenceNode]) -> None:
        """
        Initialize the pipeline with nodes.

        :param nodes: list of nodes.
        """
        self.nodes = {node.node_type: node for node in nodes}

    @classmethod
    def from_dict_config(cls, nodes_configs: list[dict[str, Any]]) -> Self:
        """
        Create pipeline from dictionary config.

        :param nodes_configs: list of dictionary config for nodes
        :return: InferencePipeline
        """
        nodes_configs_ = [InferenceNodeConfig(**cfg) for cfg in nodes_configs]
        nodes = [InferenceNode.from_config(cfg) for cfg in nodes_configs_]
        return cls(nodes)

    @classmethod
    def from_config(cls, nodes_configs: list[InferenceNodeConfig]) -> Self:
        """
        Create pipeline from config.

        :param nodes_configs: list of config for nodes
        """
        nodes = [InferenceNode.from_config(cfg) for cfg in nodes_configs]
        return cls(nodes)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with (Path(path) / "inference_config.yaml").open() as file:
            inference_dict_config = yaml.safe_load(file)
        return cls.from_dict_config(inference_dict_config["nodes_configs"])

    def predict(self, utterances: list[str]) -> list[LabelType]:
        """
        Predict the labels for the utterances.

        :param utterances: list of utterances
        :return: list of predicted labels
        """
        scores = self.nodes[NodeType.scoring.value].module.predict(utterances)
        return self.nodes[NodeType.prediction.value].module.predict(scores)  # type: ignore[return-value]

    def predict_with_metadata(self, utterances: list[str]) -> InferencePipelineOutput:
        """
        Predict the labels for the utterances with metadata.

        :param utterances: list of utterances
        :return: prediction output
        """
        scores, scores_metadata = self.nodes["scoring"].module.predict_with_metadata(utterances)
        predictions = self.nodes["prediction"].module.predict(scores)
        regexp_predictions, regexp_predictions_metadata = None, None
        if "regexp" in self.nodes:
            regexp_predictions, regexp_predictions_metadata = self.nodes["regexp"].module.predict_with_metadata(
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

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        """
        Pipeline for inference not support fit.

        :param utterances: utterances
        :param labels: labels
        """

    @classmethod
    def from_context(cls, context: Context) -> Self:
        """
        Create pipeline from context.

        :param context: context
        """
        if not context.has_saved_modules():
            config = context.optimization_info.get_inference_nodes_config()
            return cls.from_config(config)
        nodes = [
            InferenceNode(module, node_type)
            for node_type, module in context.optimization_info.get_best_modules().items()
        ]
        return cls(nodes)
