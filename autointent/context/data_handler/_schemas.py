"""Module for defining schemas for tags, intents, and utterances.

This module provides data models for utterances, intents, and tags.
"""

from pydantic import BaseModel, model_validator

from autointent.custom_types import LabelType


class Tag(BaseModel):
    """
    Represents a tag associated with intent classes.

    Tags are used to define constraints such that if two intent classes share
    a common tag, they cannot both be assigned to the same sample.
    """

    name: str
    intent_ids: list[int]


class Sample(BaseModel):
    """
    Represents a sample with an utterance and an optional label.

    :param utterance: The textual content of the sample.
    :param label: The label(s) associated with the sample. Can be a single label (integer)
                  or a list of labels (integers). Defaults to None for unlabeled samples.
    """

    utterance: str
    label: LabelType | None = None

    @model_validator(mode="after")
    def validate_sample(self) -> "Sample":
        """
        Validate the sample after model instantiation.

        This method ensures that the `label` field adheres to the expected constraints:
        - If `label` is provided, it must be a non-negative integer or a list of non-negative integers.
        - Multilabel samples must have at least one valid label.

        :raises ValueError: If the `label` field is empty for a multilabel sample
                            or contains invalid (negative) values.
        :return: The validated sample instance.
        """
        return self._validate_label()

    def _validate_label(self) -> "Sample":
        """
        Validate the `label` field of the sample.

        - Ensures that the `label` is not empty for multilabel samples.
        - Validates that all provided labels are non-negative integers.

        :raises ValueError: If the `label` is empty for a multilabel sample or
                            contains any negative values.
        :return: The validated sample instance.
        """
        if self.label is None:
            return self
        label = [self.label] if isinstance(self.label, int) else self.label
        if not label:
            message = (
                "The `label` field cannot be empty for a multilabel sample. " "Please provide at least one valid label."
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
    """Represents an intent with its metadata and regular expressions."""

    id: int
    name: str | None = None
    tags: list[str] = []
    regexp_full_match: list[str] = []
    regexp_partial_match: list[str] = []
    description: str | None = None
