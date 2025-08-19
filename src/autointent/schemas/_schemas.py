"""Module for defining schemas for tags, intents, and utterances.

This module provides data models for utterances, intents, and tags.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    model_validator,
)

from autointent.custom_types import LabelWithOOS


class Tag(BaseModel):
    """Represents a tag associated with intent classes.

    Tags are used to define constraints such that if two intent classes share
    a common tag, they cannot both be assigned to the same sample.
    """

    name: str
    intent_ids: list[int]


class TagsList(list[Tag]):
    def __init__(self, tags: list[Tag]) -> None:
        super().__init__(tags)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [v.model_dump(mode="json") for v in self]
        with path.open("w", encoding="utf-8") as file:
            json.dump(serialized, file, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "TagsList":
        """Load pydantic model from file system."""
        with path.open(encoding="utf-8") as file:
            serialized: list[dict[str, Any]] = json.load(file)
        parsed = [Tag(**t) for t in serialized]
        return cls(parsed)


class Sample(BaseModel):
    """Represents a sample with an utterance and an optional label.

    :param utterance: The textual content of the sample.
    :param label: The label(s) associated with the sample. Can be a single label (integer)
                  or a list of labels (integers). Defaults to None for unlabeled samples.
    """

    utterance: str
    label: LabelWithOOS = None

    @model_validator(mode="after")
    def validate_sample(self) -> "Sample":
        """Validate the sample after model instantiation.

        This method ensures that the `label` field adheres to the expected constraints:
        - If `label` is provided, it must be a non-negative integer or a list of non-negative integers.
        - Multilabel samples must have at least one valid label.

        :raises ValueError: If the `label` field is empty for a multilabel sample
                            or contains invalid (negative) values.
        :return: The validated sample instance.
        """
        return self._validate_label()

    def _validate_label(self) -> "Sample":
        """Validate the `label` field of the sample.

        - Ensures that the `label` is not empty for multilabel samples.
        - Validates that all provided labels are non-negative integers.

        :raises ValueError: If the `label` is empty for a multilabel sample or
                            contains any negative values.
        :return: The validated sample instance.
        """
        if self.label is None:
            return self
        if isinstance(self.label, int) and self.label < 0:
            message = (
                f"All label values must be non-negative integers. Met {self.label} "
                "Ensure that each label falls within the valid range of 0 to `n_classes - 1`."
            )
            raise ValueError(message)
        if isinstance(self.label, list):
            if len(self.label) == 0:
                message = (
                    "The `label` field cannot be empty for a multilabel sample. "
                    "Please provide at least one valid label."
                )
                raise ValueError(message)
            if any(lab not in [0, 1] for lab in self.label):
                message = "In multi-label case, all labels need to be one hot encoded."
                raise ValueError(message)
            if sum(self.label) == 0:
                message = (
                    "Found full-zero label. It must contain at least one 1. "
                    "If you wanted to define OOS sample, simply omit the label field."
                )
                raise ValueError(message)
        return self


class Intent(BaseModel):
    """Represents an intent with its metadata and regular expressions."""

    id: int
    name: str | None = None
    tags: list[str] = []
    regex_full_match: list[str] = []
    regex_partial_match: list[str] = []
    description: str | None = None
