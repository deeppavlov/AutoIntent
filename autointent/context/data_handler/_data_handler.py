"""Data Handler file."""

import logging
from pathlib import Path
from typing import TypedDict, cast

from datasets import concatenate_datasets
from transformers import set_seed

from autointent import Dataset
from autointent.custom_types import ListOfGenericLabels, Split

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

    def __init__(self, dataset: Dataset, random_seed: int = 0, split_train: bool = True) -> None:
        """
        Initialize the data handler.

        :param dataset: Training dataset.
        :param random_seed: Seed for random number generation.
        :param split_train: Perform or not splitting of train (default to split to be used in scoring and
                            threshold search).
        """
        set_seed(random_seed)

        self.dataset = dataset

        self.n_classes = self.dataset.n_classes

        self._split(random_seed, split_train)

        self.regexp_patterns = [
            RegexPatterns(
                id=intent.id,
                regexp_full_match=intent.regexp_full_match,
                regexp_partial_match=intent.regexp_partial_match,
            )
            for intent in self.dataset.intents
        ]

        self.intent_descriptions = [intent.description for intent in self.dataset.intents]
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
        Retrieve training utterances from the dataset.

        If a specific training split index is provided, retrieves utterances
        from the indexed training split. Otherwise, retrieves utterances from
        the primary training split.

        :param idx: Optional index for a specific training split.
        :return: List of training utterances.
        """
        split = f"{Split.TRAIN}_{idx}" if idx is not None else Split.TRAIN
        return cast(list[str], self.dataset[split][self.dataset.utterance_feature])

    def train_labels(self, idx: int | None = None) -> ListOfGenericLabels:
        """
        Retrieve training labels from the dataset.

        If a specific training split index is provided, retrieves labels
        from the indexed training split. Otherwise, retrieves labels from
        the primary training split.

        :param idx: Optional index for a specific training split.
        :return: List of training labels.
        """
        split = f"{Split.TRAIN}_{idx}" if idx is not None else Split.TRAIN
        return cast(ListOfGenericLabels, self.dataset[split][self.dataset.label_feature])

    def validation_utterances(self, idx: int | None = None) -> list[str]:
        """
        Retrieve validation utterances from the dataset.

        If a specific validation split index is provided, retrieves utterances
        from the indexed validation split. Otherwise, retrieves utterances from
        the primary validation split.

        :param idx: Optional index for a specific validation split.
        :return: List of validation utterances.
        """
        split = f"{Split.VALIDATION}_{idx}" if idx is not None else Split.VALIDATION
        return cast(list[str], self.dataset[split][self.dataset.utterance_feature])

    def validation_labels(self, idx: int | None = None) -> ListOfGenericLabels:
        """
        Retrieve validation labels from the dataset.

        If a specific validation split index is provided, retrieves labels
        from the indexed validation split. Otherwise, retrieves labels from
        the primary validation split.

        :param idx: Optional index for a specific validation split.
        :return: List of validation labels.
        """
        split = f"{Split.VALIDATION}_{idx}" if idx is not None else Split.VALIDATION
        return cast(ListOfGenericLabels, self.dataset[split][self.dataset.label_feature])

    def test_utterances(self, idx: int | None = None) -> list[str]:
        """
        Retrieve test utterances from the dataset.

        If a specific test split index is provided, retrieves utterances
        from the indexed test split. Otherwise, retrieves utterances from
        the primary test split.

        :param idx: Optional index for a specific test split.
        :return: List of test utterances.
        """
        split = f"{Split.TEST}_{idx}" if idx is not None else Split.TEST
        return cast(list[str], self.dataset[split][self.dataset.utterance_feature])

    def test_labels(self, idx: int | None = None) -> ListOfGenericLabels:
        """
        Retrieve test labels from the dataset.

        If a specific test split index is provided, retrieves labels
        from the indexed test split. Otherwise, retrieves labels from
        the primary test split.

        :param idx: Optional index for a specific test split.
        :return: List of test labels.
        """
        split = f"{Split.TEST}_{idx}" if idx is not None else Split.TEST
        return cast(ListOfGenericLabels, self.dataset[split][self.dataset.label_feature])

    def dump(self, filepath: str | Path) -> None:
        """
        Save the dataset splits and intents to a JSON file.

        :param filepath: The path to the file where the JSON data will be saved.
        """
        self.dataset.to_json(filepath)

    def _split(self, random_seed: int, split_train: bool) -> None:
        has_validation_split = any(split.startswith(Split.VALIDATION) for split in self.dataset)

        if split_train and Split.TRAIN in self.dataset:
            self._split_train(random_seed)

        if Split.TEST not in self.dataset:
            test_size = 0.1 if has_validation_split else 0.2
            self._split_test(test_size, random_seed)

        if not has_validation_split:
            self._split_validation_from_train(random_seed)
        elif Split.VALIDATION in self.dataset:
            self._split_validation(random_seed)

        for split in self.dataset:
            n_classes_split = self.dataset.get_n_classes(split)
            if n_classes_split != self.n_classes:
                message = (
                    f"Number of classes in split '{split}' doesn't match initial number of classes "
                    f"({n_classes_split} != {self.n_classes})"
                )
                raise ValueError(message)

    def _split_train(self, random_seed: int) -> None:
        """
        Split on two sets.

        One is for scoring node optimizaton, one is for decision node.
        """
        self.dataset[f"{Split.TRAIN}_0"], self.dataset[f"{Split.TRAIN}_1"] = split_dataset(
            self.dataset,
            split=Split.TRAIN,
            test_size=0.5,
            random_seed=random_seed,
            allow_oos_in_train=False,  # only train data for decision node should contain OOS
        )
        self.dataset.pop(Split.TRAIN)

    def _split_validation(self, random_seed: int) -> None:
        """
        Split on two sets.

        One is for scoring node optimizaton, one is for decision node.
        """
        self.dataset[f"{Split.VALIDATION}_0"], self.dataset[f"{Split.VALIDATION}_1"] = split_dataset(
            self.dataset,
            split=Split.VALIDATION,
            test_size=0.5,
            random_seed=random_seed,
            allow_oos_in_train=False,  # only val data for decision node should contain OOS
        )
        self.dataset.pop(Split.VALIDATION)

    def _split_validation_from_test(self, random_seed: int) -> None:
        self.dataset[Split.TEST], self.dataset[Split.VALIDATION] = split_dataset(
            self.dataset,
            split=Split.TEST,
            test_size=0.5,
            random_seed=random_seed,
            allow_oos_in_train=True,  # both test and validation splits can contain OOS
        )

    def _split_validation_from_train(self, random_seed: int) -> None:
        if Split.TRAIN in self.dataset:
            self.dataset[Split.TRAIN], self.dataset[Split.VALIDATION] = split_dataset(
                self.dataset,
                split=Split.TRAIN,
                test_size=0.2,
                random_seed=random_seed,
                allow_oos_in_train=True,
            )
        else:
            for idx in range(2):
                self.dataset[f"{Split.TRAIN}_{idx}"], self.dataset[f"{Split.VALIDATION}_{idx}"] = split_dataset(
                    self.dataset,
                    split=f"{Split.TRAIN}_{idx}",
                    test_size=0.2,
                    random_seed=random_seed,
                    allow_oos_in_train=idx == 1,  # for decision node it's ok to have oos in train
                )

    def _split_test(self, test_size: float, random_seed: int) -> None:
        """Obtain test set from train."""
        self.dataset[f"{Split.TRAIN}_0"], self.dataset[f"{Split.TEST}_0"] = split_dataset(
            self.dataset,
            split=f"{Split.TRAIN}_0",
            test_size=test_size,
            random_seed=random_seed,
        )
        self.dataset[f"{Split.TRAIN}_1"], self.dataset[f"{Split.TEST}_1"] = split_dataset(
            self.dataset,
            split=f"{Split.TRAIN}_1",
            test_size=test_size,
            random_seed=random_seed,
            allow_oos_in_train=True,
        )
        self.dataset[Split.TEST] = concatenate_datasets(
            [self.dataset[f"{Split.TEST}_0"], self.dataset[f"{Split.TEST}_1"]],
        )
        self.dataset.pop(f"{Split.TEST}_0")
        self.dataset.pop(f"{Split.TEST}_1")
