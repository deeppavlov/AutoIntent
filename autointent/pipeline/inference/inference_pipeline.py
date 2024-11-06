from typing import Any

from hydra.utils import instantiate
from pydantic import BaseModel

from autointent.configs.inference_pipeline import InferencePipelineConfig
from autointent.custom_types import LabelType
from autointent.nodes.inference import InferenceNode


class InferencePipelineUtteranceOutput(BaseModel):
    utterance: str
    prediction: LabelType
    score: list[float]
    score_metadata: Any


class InferencePipelineOutput(BaseModel):
    predictions: list[LabelType]
    utterances: list[InferencePipelineUtteranceOutput] | None = None


class InferencePipeline:
    def __init__(self, nodes: list[InferenceNode]) -> None:
        self.nodes = {node.node_info.node_type: node for node in nodes}

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> "InferencePipeline":
        return instantiate(InferencePipelineConfig, **config)  # type: ignore[no-any-return]

    def predict(self, utterances: list[str]) -> InferencePipelineOutput:
        scores = self.nodes["scoring"].module.predict(utterances)
        predictions = self.nodes["prediction"].module.predict(scores)
        return InferencePipelineOutput(predictions=predictions)

    def predict_with_metadata(self, utterances: list[str]) -> InferencePipelineOutput:
        scores, scores_metadata = self.nodes["scoring"].module.predict_with_metadata(utterances)
        predictions = self.nodes["prediction"].module.predict(scores)

        outputs = []
        for idx, utterance in enumerate(utterances):
            output = InferencePipelineUtteranceOutput(
                utterance=utterance,
                prediction=predictions[idx],
                score=scores[idx],
                score_metadata=scores_metadata[idx] if scores_metadata is not None else None,
            )
            outputs.append(output)

        return InferencePipelineOutput(predictions=predictions, utterances=outputs)

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        pass
