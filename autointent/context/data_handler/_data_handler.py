"""Data Handler file."""

import logging
from collections.abc import Generator
from typing import TypedDict, cast

from datasets import concatenate_datasets
from transformers import set_seed

from autointent import Dataset
from autointent.custom_types import ListOfGenericLabels, ListOfLabels, Split, ValidationScheme

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


class DataHandler:  # TODO rename to Validator
    """Data handler class."""

    def __init__(
        self,
        dataset: Dataset,
        scheme: ValidationScheme = "ho",
        split_train: bool = True,
        random_seed: int = 0,
        n_folds: int = 3,
    ) -> None:
        """
        Initialize the data handler.

        :param dataset: Training dataset.
        :param random_seed: Seed for random number generation.
        :param split_train: Perform or not splitting of train (default to split to be used in scoring and
                            threshold search).
        """
        set_seed(random_seed)
        self.random_seed = random_seed

        self.dataset = dataset

        self.n_classes = self.dataset.n_classes
        self.scheme = scheme
        self.n_folds = n_folds

        if scheme == "ho":
            self._split_ho(split_train)
        elif scheme == "cv":
            self._split_cv()

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

    def train_labels_folded(self) -> list[ListOfGenericLabels]:
        return [self.train_labels(j) for j in range(self.n_folds)]

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

    def validation_iterator(self) -> Generator[tuple[list[str], ListOfLabels, list[str], ListOfLabels]]:
        if self.scheme == "ho":
            msg = "Cannot call cross-validation on hold-out DataHandler"
            raise RuntimeError(msg)

        for j in range(self.n_folds):
            val_utterances = self.train_utterances(j)
            val_labels = self.train_labels(j)
            train_folds = [i for i in range(self.n_folds) if i != j]
            train_utterances = [ut for i_fold in train_folds for ut in self.train_utterances(i_fold)]
            train_labels = [lab for i_fold in train_folds for lab in self.train_labels(i_fold)]

            # filter out all OOS samples from train
            train_utterances = [ut for ut, lab in zip(train_utterances, train_labels, strict=True) if lab is not None]
            train_labels = [lab for lab in train_labels if lab is not None]
            yield train_utterances, train_labels, val_utterances, val_labels  # type: ignore[misc]

    def _split_ho(self, split_train: bool) -> None:
        has_validation_split = any(split.startswith(Split.VALIDATION) for split in self.dataset)

        if split_train and Split.TRAIN in self.dataset:
            self._split_train()

        if Split.TEST not in self.dataset:
            test_size = 0.1 if has_validation_split else 0.2
            self._split_test(test_size)

        if not has_validation_split:
            self._split_validation_from_train()
        elif Split.VALIDATION in self.dataset:
            self._split_validation()

        for split in self.dataset:
            n_classes_split = self.dataset.get_n_classes(split)
            if n_classes_split != self.n_classes:
                message = (
                    f"Number of classes in split '{split}' doesn't match initial number of classes "
                    f"({n_classes_split} != {self.n_classes})"
                )
                raise ValueError(message)

    def _split_train(self) -> None:
        """
        Split on two sets.

        One is for scoring node optimizaton, one is for decision node.
        """
        self.dataset[f"{Split.TRAIN}_0"], self.dataset[f"{Split.TRAIN}_1"] = split_dataset(
            self.dataset,
            split=Split.TRAIN,
            test_size=0.5,
            random_seed=self.random_seed,
            allow_oos_in_train=False,  # only train data for decision node should contain OOS
        )
        self.dataset.pop(Split.TRAIN)

    def _split_validation(self) -> None:
        """
        Split on two sets.

        One is for scoring node optimizaton, one is for decision node.
        """
        self.dataset[f"{Split.VALIDATION}_0"], self.dataset[f"{Split.VALIDATION}_1"] = split_dataset(
            self.dataset,
            split=Split.VALIDATION,
            test_size=0.5,
            random_seed=self.random_seed,
            allow_oos_in_train=False,  # only val data for decision node should contain OOS
        )
        self.dataset.pop(Split.VALIDATION)

    def _split_validation_from_test(self) -> None:
        self.dataset[Split.TEST], self.dataset[Split.VALIDATION] = split_dataset(
            self.dataset,
            split=Split.TEST,
            test_size=0.5,
            random_seed=self.random_seed,
            allow_oos_in_train=True,  # both test and validation splits can contain OOS
        )

    def _split_cv(self) -> None:
        extra_splits = [split_name for split_name in self.dataset if split_name not in [Split.TRAIN, Split.TEST]]
        if extra_splits:
            self.dataset[Split.TRAIN] = concatenate_datasets(
                [self.dataset.pop(split_name) for split_name in extra_splits]
            )

        if Split.TEST not in self.dataset:
            self.dataset[Split.TRAIN], self.dataset[Split.TEST] = split_dataset(
                self.dataset, split=Split.TRAIN, test_size=0.2, random_seed=self.random_seed, allow_oos_in_train=True
            )

        for j in range(self.n_folds - 1):
            self.dataset[Split.TRAIN], self.dataset[f"{Split.TRAIN}_{j}"] = split_dataset(
                self.dataset,
                split=Split.TRAIN,
                test_size=1 / (self.n_folds - j),
                random_seed=self.random_seed,
                allow_oos_in_train=True,
            )
        self.dataset[f"{Split.TRAIN}_{self.n_folds-1}"] = self.dataset.pop(Split.TRAIN)

    def _split_validation_from_train(self) -> None:
        if Split.TRAIN in self.dataset:
            self.dataset[Split.TRAIN], self.dataset[Split.VALIDATION] = split_dataset(
                self.dataset,
                split=Split.TRAIN,
                test_size=0.2,
                random_seed=self.random_seed,
                allow_oos_in_train=True,
            )
        else:
            for idx in range(2):
                self.dataset[f"{Split.TRAIN}_{idx}"], self.dataset[f"{Split.VALIDATION}_{idx}"] = split_dataset(
                    self.dataset,
                    split=f"{Split.TRAIN}_{idx}",
                    test_size=0.2,
                    random_seed=self.random_seed,
                    allow_oos_in_train=idx == 1,  # for decision node it's ok to have oos in train
                )

    def _split_test(self, test_size: float) -> None:
        """Obtain test set from train."""
        self.dataset[f"{Split.TRAIN}_0"], self.dataset[f"{Split.TEST}_0"] = split_dataset(
            self.dataset,
            split=f"{Split.TRAIN}_0",
            test_size=test_size,
            random_seed=self.random_seed,
        )
        self.dataset[f"{Split.TRAIN}_1"], self.dataset[f"{Split.TEST}_1"] = split_dataset(
            self.dataset,
            split=f"{Split.TRAIN}_1",
            test_size=test_size,
            random_seed=self.random_seed,
            allow_oos_in_train=True,
        )
        self.dataset[Split.TEST] = concatenate_datasets(
            [self.dataset[f"{Split.TEST}_0"], self.dataset[f"{Split.TEST}_1"]],
        )
        self.dataset.pop(f"{Split.TEST}_0")
        self.dataset.pop(f"{Split.TEST}_1")

    def prepare_for_refit(self) -> None:
        if self.scheme == "ho":
            return

        train_folds = [split_name for split_name in self.dataset if split_name.startswith(Split.TRAIN)]
        self.dataset[Split.TRAIN] = concatenate_datasets([self.dataset.pop(name) for name in train_folds])

        self.dataset[f"{Split.TRAIN}_0"], self.dataset[f"{Split.TRAIN}_1"] = split_dataset(
            self.dataset,
            split=Split.TRAIN,
            test_size=0.5,
            random_seed=self.random_seed,
            allow_oos_in_train=False,
        )

        self.dataset.pop(Split.TRAIN)
