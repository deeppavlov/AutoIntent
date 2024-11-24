"""Module for defining schemas for datasets, intents, and utterances.

This module provides data models for utterances, intents, and datasets.
It includes functionality for dataset transformation, label encoding,
and interaction with Hugging Face datasets.
"""

from enum import Enum
from functools import cached_property
from typing import Any

import datasets
from pydantic import BaseModel
from typing_extensions import Self

from autointent.custom_types import LabelType


class UtteranceType(str, Enum):
    """Enumeration of possible utterance types."""

    oos = "oos"  # Out of Scope
    multilabel = "multilabel"
    multiclass = "multiclass"


class DatasetType(str, Enum):
    """Enumeration of possible dataset types."""

    multilabel = "multilabel"
    multiclass = "multiclass"


class Utterance(BaseModel):
    """Represents an utterance with text and an optional label."""

    text: str
    label: LabelType | None = None

    def one_hot_label(self, n_classes: int) -> LabelType:
        """
        Convert the label into a one-hot encoded vector.

        :param n_classes: Total number of classes.
        :return: A list representing the one-hot encoded label.
        :raises ValueError: If the label is None (Out of Scope).
        """
        encoding = [0] * n_classes
        label = [self.label] if isinstance(self.label, int) else self.label
        if label is None:
            msg = "Cannot one-hot encode OOS utterance"
            raise ValueError(msg)
        for idx in label:
            encoding[idx] = 1
        return encoding

    @cached_property
    def type(self) -> UtteranceType:
        """
        Determine the type of the utterance based on its label.

        :return: The type of the utterance (oos, multiclass, or multilabel).
        """
        if self.label is None:
            return UtteranceType.oos
        if isinstance(self.label, int):
            return UtteranceType.multiclass
        return UtteranceType.multilabel

    @cached_property
    def oos(self) -> bool:
        """
        Check if the utterance is Out of Scope (OOS).

        :return: True if the utterance is OOS, False otherwise.
        """
        return self.label is None

    def to_multilabel(self) -> "Utterance":
        """
        Convert the utterance to a multilabel format.

        :return: A new `Utterance` object in multilabel format.
        """
        if self.type in {UtteranceType.multilabel, UtteranceType.oos}:
            return self
        return Utterance(text=self.text, label=[self.label])


class Intent(BaseModel):
    """Represents an intent with its metadata and regular expressions."""

    id: int
    name: str | None = None
    tags: list[str] = []
    regexp_full_match: list[str] = []
    regexp_partial_match: list[str] = []
    description: str | None = None


class Dataset(BaseModel):
    """Represents a dataset consisting of utterances and intents."""

    utterances: list[Utterance]
    intents: list[Intent] = []

    @cached_property
    def type(self) -> DatasetType:
        """
        Determine the type of the dataset (multiclass or multilabel).

        :return: The type of the dataset.
        """
        if all(utterance.type in {UtteranceType.multiclass, UtteranceType.oos} for utterance in self.utterances):
            return DatasetType.multiclass
        return DatasetType.multilabel  # TODO: Add proper dataset type validation

    @cached_property
    def n_classes(self) -> int:
        """
        Calculate the number of unique classes in the dataset.

        :return: The number of classes.
        """
        classes = set()
        for utterance in self.utterances:
            if utterance.oos:
                continue
            if self.type == DatasetType.multiclass:
                classes.add(utterance.label)
            elif isinstance(utterance.label, list):
                for label in utterance.label:
                    classes.add(label)
        return len(classes)

    def to_multilabel(self) -> "Dataset":
        """
        Convert the dataset to a multilabel format.

        :return: A new `Dataset` object in multilabel format.
        """
        return Dataset(utterances=[utterance.to_multilabel() for utterance in self.utterances], intents=self.intents)

    @classmethod
    def from_datasets(
        cls,
        dataset_name: str,
        split: str = "train",
        utterances_kwargs: dict[str, Any] | None = None,
        intents_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        """
        Load a dataset from the Hugging Face Hub and create a `Dataset` object.

        :param dataset_name: Name of the dataset on the Hugging Face Hub.
        :param split: Split of the dataset to load (e.g., "train").
        :param utterances_kwargs: Additional arguments for loading utterances.
        :param intents_kwargs: Additional arguments for loading intents.
        :return: A `Dataset` object with loaded utterances and intents.
        """
        configs = datasets.get_dataset_config_names(dataset_name)

        utterances = []
        intents = []
        if "utterances" in configs:
            utterance_ds = datasets.load_dataset(
                dataset_name, name="utterances", split=split, **(utterances_kwargs or {})
            )
            utterances = [Utterance(**item) for item in utterance_ds]
        if "intents" in configs:
            intents_ds = datasets.load_dataset(dataset_name, name="intents", split=split, **(intents_kwargs or {}))
            intents = [Intent(**item) for item in intents_ds]
        return cls(utterances=utterances, intents=intents)

    def push_to_hub(self, dataset_name: str, split: str = "train") -> None:
        """
        Push the dataset to the Hugging Face Hub.

        Uploads the utterances and intents to the specified dataset on the Hub.

        :param dataset_name: Name of the dataset on the Hugging Face Hub.
        :param split: Split of the dataset to upload (e.g., "train").
        """
        utterances_ds = datasets.Dataset.from_list([utterance.model_dump() for utterance in self.utterances])
        intents_ds = datasets.Dataset.from_list([intent.model_dump() for intent in self.intents])
        utterances_ds.push_to_hub(dataset_name, config_name="utterances", split=split)
        intents_ds.push_to_hub(dataset_name, config_name="intents", split=split)
