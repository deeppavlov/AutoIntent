"""Data Handler file."""

import logging
from typing import Any, TypedDict, cast

from transformers import set_seed

from autointent.custom_types import LabelType

from .dataset import Dataset, Split
from .stratification import split_dataset

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

        if Split.TEST not in self.dataset:
            logger.info("Spltting dataset into train and test splits")
            self.dataset = split_dataset(self.dataset, random_seed=random_seed)

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
        return self.dataset.multilabel

    @property
    def n_classes(self) -> int:
        return self.dataset.n_classes

    @property
    def train_utterances(self) -> list[str]:
        return cast(list[str], self.dataset[Split.TRAIN][self.dataset.utterance_feature])

    @property
    def train_labels(self) -> list[LabelType]:
        return cast(list[LabelType], self.dataset[Split.TRAIN][self.dataset.label_feature])

    @property
    def test_utterances(self) -> list[str]:
        return cast(list[str], self.dataset[Split.TEST][self.dataset.utterance_feature])

    @property
    def test_labels(self) -> list[LabelType]:
        return cast(list[LabelType], self.dataset[Split.TEST][self.dataset.label_feature])

    @property
    def oos_utterances(self) -> list[str]:
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
