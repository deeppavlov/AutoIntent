from pydantic import BaseModel, model_validator
from typing_extensions import Self

from autointent.custom_types import LabelType


class Tag(BaseModel):
    name: str
    intent_ids: list[int]


class Sample(BaseModel):
    utterance: str
    label: LabelType | None = None

    @model_validator(mode="after")
    def validate_sample(self) -> Self:
        return self._validate_label()

    def _validate_label(self) -> Self:
        if self.label is None:
            return self
        label = [self.label] if isinstance(self.label, int) else self.label
        if not label:
            message = (
                "The `label` field cannot be empty for a multilabel sample. "
                "Please provide at least one valid label."
            )
            raise ValueError(message)
        if any(label_ < 0 for label_ in label):
            message = (
                "All label values must be non-negative integers. "
                "Ensure that each label falls within the valid range of 0 to `n_classes - 1`."
            )
            raise ValueError(message)
        return self


class Intent(BaseModel):
    id: int
    name: str | None = None
    tags: list[str] = []
    regexp_full_match: list[str] = []
    regexp_partial_match: list[str] = []
    description: str | None = None
