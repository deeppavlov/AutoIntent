from typing import Any

from pydantic import BaseModel

from autointent.custom_types import LabelWithOOS, ListOfLabels, ListOfLabelsWithOOS


class InferencePipelineUtteranceOutput(BaseModel):
    """Output of the inference pipeline for a single utterance."""

    utterance: str
    prediction: LabelWithOOS
    regexp_prediction: LabelWithOOS
    regexp_prediction_metadata: Any
    score: list[float]
    score_metadata: Any


class InferencePipelineOutput(BaseModel):
    """Output of the inference pipeline."""

    predictions: ListOfLabelsWithOOS
    regexp_predictions: ListOfLabels | None = None
    utterances: list[InferencePipelineUtteranceOutput] | None = None
