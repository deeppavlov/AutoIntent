from typing import Any

from pydantic import BaseModel

from autointent.custom_types import LabelType


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
