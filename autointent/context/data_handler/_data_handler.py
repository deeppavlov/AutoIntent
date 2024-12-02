"""Data Handler file."""

import logging
from typing import Any, TypedDict, cast

from transformers import set_seed

from autointent.custom_types import LabelType

from ._dataset import Dataset, Split
from ._stratification import split_dataset

logger = logging.getLogger(__name__)


class RegexPatterns(TypedDict):
    """Regex patterns for each intent class."""

    id: int
    """Intent class id."""
    regexp_full_match: list[str]
    """Full match regex patterns."""
    regexp_partial_match: list[str]
    """Partial match regex patterns."""


class DataHandler:
    """Data handler class."""

    def __init__(
        self,
        dataset: Dataset,
        force_multilabel: bool = False,
        random_seed: int = 0,
    ) -> None:
        """
        Initialize the data handler.

        :param dataset: Training dataset.
        :param force_multilabel: If True, force the dataset to be multilabel.
        :param random_seed: Seed for random number generation.
        """
        set_seed(random_seed)

        self.dataset = dataset
        if force_multilabel:
            self.dataset = self.dataset.to_multilabel()
        if self.dataset.multilabel:
            self.dataset = self.dataset.encode_labels()

        self.n_classes = self.dataset.n_classes

        if Split.TEST not in self.dataset:
            self.dataset[Split.TRAIN], self.dataset[Split.TEST] = split_dataset(
                self.dataset,
                split=Split.TRAIN,
                test_size=0.2,
                random_seed=random_seed,
            )

        self.dataset[f"{Split.TRAIN}_0"], self.dataset[f"{Split.TRAIN}_1"] = split_dataset(
            self.dataset,
            split=Split.TRAIN,
            test_size=0.5,
            random_seed=random_seed,
        )
        self.dataset.pop(Split.TRAIN)

        for idx in range(2):
            self.dataset[f"{Split.TRAIN}_{idx}"], self.dataset[f"{Split.VALIDATION}_{idx}"] = split_dataset(
                self.dataset,
                split=f"{Split.TRAIN}_{idx}",
                test_size=0.2,
                random_seed=random_seed,
            )

        for split in self.dataset:
            if split == Split.OOS:
                continue
            n_classes_split = self.dataset.get_n_classes(split)
            if n_classes_split != self.n_classes:
                message = (
                    f"Number of classes in split '{split}' doesn't match initial number of classes "
                    f"({n_classes_split} != {self.n_classes})"
                )
                raise ValueError(message)

        self.regexp_patterns = [
            RegexPatterns(
                id=intent.id,
                regexp_full_match=intent.regexp_full_match,
                regexp_partial_match=intent.regexp_partial_match,
            )
            for intent in self.dataset.intents
        ]

        self.intent_descriptions = [intent.name for intent in self.dataset.intents]
        self.tags = self.dataset.get_tags()

        self._logger = logger

    @property
    def multilabel(self) -> bool:
        """
        Check if the dataset is multilabel.

        :return: True if the dataset is multilabel, False otherwise.
        """
        return self.dataset.multilabel

    def train_utterances(self, idx: int | None = None) -> list[str]:
        """
        Get the training utterances.

        :return: List of training utterances.
        """
        split = f"{Split.TRAIN}_{idx}" if idx is not None else Split.TRAIN
        return cast(list[str], self.dataset[split][self.dataset.utterance_feature])

    def train_labels(self, idx: int | None = None) -> list[LabelType]:
        """
        Get the training labels.

        :return: List of training labels.
        """
        split = f"{Split.TRAIN}_{idx}" if idx is not None else Split.TRAIN
        return cast(list[LabelType], self.dataset[split][self.dataset.label_feature])

    def validation_utterances(self, idx: int | None = None) -> list[str]:
        """
        Get the validation utterances.

        :return: List of validation utterances.
        """
        split = f"{Split.VALIDATION}_{idx}" if idx is not None else Split.VALIDATION
        return cast(list[str], self.dataset[split][self.dataset.utterance_feature])

    def validation_labels(self, idx: int | None = None) -> list[LabelType]:
        """
        Get the validatio labels.

        :return: List of validatio labels.
        """
        split = f"{Split.VALIDATION}_{idx}" if idx is not None else Split.VALIDATION
        return cast(list[LabelType], self.dataset[split][self.dataset.label_feature])

    def test_utterances(self, idx: int | None = None) -> list[str]:
        """
        Get the test utterances.

        :return: List of test utterances.
        """
        split = f"{Split.TEST}_{idx}" if idx is not None else Split.TEST
        return cast(list[str], self.dataset[split][self.dataset.utterance_feature])

    def test_labels(self, idx: int | None = None) -> list[LabelType]:
        """
        Get the test labels.

        :return: List of test labels.
        """
        split = f"{Split.TEST}_{idx}" if idx is not None else Split.TEST
        return cast(list[LabelType], self.dataset[split][self.dataset.label_feature])

    def oos_utterances(self) -> list[str]:
        """
        Get the out-of-scope utterances.

        :return: List of out-of-scope utterances if available, otherwise an empty list.
        """
        if self.has_oos_samples():
            return cast(list[str], self.dataset[Split.OOS][self.dataset.utterance_feature])
        return []

    def has_oos_samples(self) -> bool:
        """
        Check if there are out-of-scope samples.

        :return: True if there are out-of-scope samples.
        """
        return Split.OOS in self.dataset

    def dump(self) -> dict[str, list[dict[str, Any]]]:
        """
        Dump the dataset splits.

        :return: Dataset dump.
        """
        return self.dataset.dump()
