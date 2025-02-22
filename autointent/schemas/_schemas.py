"""Module for defining schemas for tags, intents, and utterances.

This module provides data models for utterances, intents, and tags.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    Field,
    NonNegativeFloat,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self

from autointent.custom_types import LabelWithOOS


class Tag(BaseModel):
    """
    Represents a tag associated with intent classes.

    Tags are used to define constraints such that if two intent classes share
    a common tag, they cannot both be assigned to the same sample.
    """

    name: str
    intent_ids: list[int]


class TagsList(list[Tag]):
    def __init__(self, tags: list[Tag]) -> None:
        super().__init__(tags)

    def dump(self, path: Path) -> None:
        serialized = [v.model_dump(mode="json") for v in self]
        with path.open("w") as file:
            json.dump(serialized, file, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "TagsList":
        """Load pydantic model from file system."""
        with path.open() as file:
            serialized: list[dict[str, Any]] = json.load(file)
        parsed = [Tag(**t) for t in serialized]
        return cls(parsed)


class Sample(BaseModel):
    """
    Represents a sample with an utterance and an optional label.

    :param utterance: The textual content of the sample.
    :param label: The label(s) associated with the sample. Can be a single label (integer)
                  or a list of labels (integers). Defaults to None for unlabeled samples.
    """

    utterance: str
    label: LabelWithOOS = None

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


class ModelConfig(BaseModel):
    batch_size: PositiveInt = Field(32, description="Batch size for model inference.")
    max_length: PositiveInt | None = Field(None, description="Maximum length of input sequences.")


class LLMConfig(ModelConfig):
    temperature: NonNegativeFloat | None = Field(None, description="Temperature for sampling from the model.")
    base_url: AnyHttpUrl | None = Field(..., description="Base URL for the model API.")
    token: str | None = Field(..., description="API token for the model.")
    extra_body: dict[str, Any] | None = Field(None, description="Extra body for the model API.")


class STModelConfig(ModelConfig):
    model_name: str = Field(..., description="Name of the hugging face model.")
    device: str | None = Field(None, description="Torch notation for CPU or CUDA.")

    @classmethod
    def from_search_config(cls, values: dict[str, Any] | str | BaseModel) -> Self:
        """Validate the model configuration.

        :param values: Model configuration values. If a string is provided, it is converted to a dictionary.
        """
        if isinstance(values, BaseModel):
            return values  # type: ignore[return-value]
        if isinstance(values, str):
            return cls(model_name=values)
        return cls(**values)


class TaskTypeEnum(Enum):
    """Enum for different types of prompts."""

    default = "default"
    classification = "classification"
    cluster = "cluster"
    query = "query"
    passage = "passage"
    sts = "sts"


class EmbedderConfig(STModelConfig):
    default_prompt: str | None = Field(
        None, description="Default prompt for the model. This is used when no task specific prompt is not provided."
    )
    classifier_prompt: str | None = Field(None, description="Prompt for classifier.")
    cluster_prompt: str | None = Field(None, description="Prompt for clustering.")
    sts_prompt: str | None = Field(None, description="Prompt for finding most similar sentences.")
    query_prompt: str | None = Field(None, description="Prompt for query.")
    passage_prompt: str | None = Field(None, description="Prompt for passage.")

    def get_prompt_config(self) -> dict[str, str] | None:
        """Get the prompt config for the given prompt type.

        :return: The prompt config for the given prompt type.
        """
        prompts = {}
        if self.default_prompt:
            prompts[TaskTypeEnum.default.value] = self.default_prompt
        if self.classifier_prompt:
            prompts[TaskTypeEnum.classification.value] = self.classifier_prompt
        if self.cluster_prompt:
            prompts[TaskTypeEnum.cluster.value] = self.cluster_prompt
        if self.query_prompt:
            prompts[TaskTypeEnum.query.value] = self.query_prompt
        if self.passage_prompt:
            prompts[TaskTypeEnum.passage.value] = self.passage_prompt
        if self.sts_prompt:
            prompts[TaskTypeEnum.sts.value] = self.sts_prompt
        return prompts if len(prompts) > 0 else None

    def get_prompt_type(self, prompt_type: TaskTypeEnum | str | None) -> str | None:  # noqa: PLR0911
        """Get the prompt type for the given task type.

        :param prompt_type: Task type for which to get the prompt.

        :return: The prompt for the given task type.
        """
        if prompt_type is None:
            return self.default_prompt
        if prompt_type == TaskTypeEnum.classification:
            return self.classifier_prompt
        if prompt_type == TaskTypeEnum.cluster:
            return self.cluster_prompt
        if prompt_type == TaskTypeEnum.query:
            return self.query_prompt
        if prompt_type == TaskTypeEnum.passage:
            return self.passage_prompt
        if prompt_type == TaskTypeEnum.sts:
            return self.sts_prompt
        if prompt_type == TaskTypeEnum.default:
            return self.default_prompt
        return None

    use_cache: bool = Field(False, description="Whether to use embeddings caching.")


class CrossEncoderConfig(STModelConfig):
    train_head: bool = Field(
        False, description="Whether to train the head of the model. If False, LogReg will be trained."
    )
