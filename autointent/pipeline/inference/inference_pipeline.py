from typing import Any

from hydra.utils import instantiate
from pydantic import BaseModel

from autointent.configs.inference_pipeline import InferencePipelineConfig
from autointent.custom_types import LabelType, NodeType
from autointent.nodes.inference import InferenceNode


class InferencePipelineUtteranceOutput(BaseModel):
    utterance: str
    prediction: LabelType
    regexp_prediction: LabelType
    regexp_prediction_metadata: Any
    score: list[float]
    score_metadata: Any


class InferencePipelineOutput(BaseModel):
    predictions: list[LabelType]
    regexp_predictions: list[LabelType] | None = None
    utterances: list[InferencePipelineUtteranceOutput] | None = None


class InferencePipeline:
    def __init__(self, nodes: list[InferenceNode]) -> None:
        self.nodes = {node.node_info.node_type: node for node in nodes}

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> "InferencePipeline":
        return instantiate(InferencePipelineConfig, **config)  # type: ignore[no-any-return]

    def predict(self, utterances: list[str]) -> list[LabelType]:
        scores = self.nodes[NodeType.scoring].module.predict(utterances)
        predictions = self.nodes[NodeType.prediction].module.predict(scores)  # type: ignore[return-value]

        regexp_predictions = None
        if "regexp" in self.nodes:
            regexp_predictions = self.nodes["regexp"].module.predict(utterances)
        return InferencePipelineOutput(predictions=predictions, regexp_predictions=regexp_predictions)

    def predict_with_metadata(self, utterances: list[str]) -> InferencePipelineOutput:
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

        return InferencePipelineOutput(predictions=predictions, utterances=outputs)

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        pass
