"""Data Handler file."""

import logging
from collections.abc import Generator
from typing import cast

from datasets import concatenate_datasets

from autointent import Dataset
from autointent.configs import DataConfig
from autointent.custom_types import FloatFromZeroToOne, ListOfGenericLabels, ListOfLabels, Split
from autointent.schemas import Tag

from ._stratification import create_few_shot_split, split_dataset

logger = logging.getLogger(__name__)


class DataHandler:
    """Convenient wrapper for :py:class:`autointent.Dataset`.

    Performs splitting of the wrapped dataset when instantiated.
    """

    dataset: Dataset
    """Wrapped dataset."""
    config: DataConfig
    """Configuration used for instantiation."""

    def __init__(
        self,
        dataset: Dataset,
        config: DataConfig | None = None,
        random_seed: int | None = 0,
    ) -> None:
        """Initialize the data handler.

        Args:
            dataset: Training dataset.
            config: Configuration object
            random_seed: Seed for random number generation.
        """
        self._seed = random_seed

        self.dataset = dataset
        self.config = config if config is not None else DataConfig()

        self._n_classes = self.dataset.n_classes

        if self.config.scheme == "ho":
            self._split_ho(
                self.config.separation_ratio,
                self.config.validation_size,
                self.config.is_few_shot_train,
                self.config.examples_per_intent,
            )
        elif self.config.scheme == "cv":
            self._split_cv(self.config.is_few_shot_train, self.config.examples_per_intent)

        self._logger = logger

    @property
    def intent_descriptions(self) -> list[str | None]:
        """String descriptions for all intents."""
        return [intent.description for intent in self.dataset.intents]

    @property
    def tags(self) -> list[Tag]:
        """Tags associated with intents.

        Tagging is an experimental feature that is not guaranteed to work.
        """
        return self.dataset.get_tags()

    @property
    def multilabel(self) -> bool:
        """Check if the dataset is multilabel."""
        return self.dataset.multilabel

    def _choose_split(self, split_name: str, idx: int | None = None) -> str:
        if idx is not None:
            split = f"{split_name}_{idx}"
            if split not in self.dataset:
                split = split_name
        else:
            split = split_name
        return split

    def train_utterances(self, idx: int | None = None) -> list[str]:
        """Retrieve training utterances from the dataset.

        If a specific training split index is provided, retrieves utterances
        from the indexed training split. Otherwise, retrieves utterances from
        the primary training split.

        Args:
            idx: Optional index for a specific training split.
        """
        split = self._choose_split(Split.TRAIN, idx)
        return cast(list[str], self.dataset[split][self.dataset.utterance_feature])

    def train_labels(self, idx: int | None = None) -> ListOfGenericLabels:
        """Retrieve training labels from the dataset.

        If a specific training split index is provided, retrieves labels
        from the indexed training split. Otherwise, retrieves labels from
        the primary training split.

        Args:
            idx: Optional index for a specific training split.
        """
        split = self._choose_split(Split.TRAIN, idx)
        return cast(ListOfGenericLabels, self.dataset[split][self.dataset.label_feature])

    def train_labels_folded(self) -> list[ListOfGenericLabels]:
        """Retrieve train labels fold by fold."""
        return [self.train_labels(j) for j in range(self.config.n_folds)]

    def validation_utterances(self, idx: int | None = None) -> list[str]:
        """Retrieve validation utterances from the dataset.

        If a specific validation split index is provided, retrieves utterances
        from the indexed validation split. Otherwise, retrieves utterances from
        the primary validation split.

        Args:
            idx: Optional index for a specific validation split.
        """
        split = self._choose_split(Split.VALIDATION, idx)
        return cast(list[str], self.dataset[split][self.dataset.utterance_feature])

    def validation_labels(self, idx: int | None = None) -> ListOfGenericLabels:
        """Retrieve validation labels from the dataset.

        If a specific validation split index is provided, retrieves labels
        from the indexed validation split. Otherwise, retrieves labels from
        the primary validation split.

        Args:
            idx: Optional index for a specific validation split.
        """
        split = self._choose_split(Split.VALIDATION, idx)
        return cast(ListOfGenericLabels, self.dataset[split][self.dataset.label_feature])

    def test_utterances(self) -> list[str] | None:
        """Retrieve test utterances from the dataset."""
        if Split.TEST not in self.dataset:
            return None
        return cast(list[str], self.dataset[Split.TEST][self.dataset.utterance_feature])

    def test_labels(self) -> ListOfGenericLabels:
        """Retrieve test labels from the dataset."""
        return cast(ListOfGenericLabels, self.dataset[Split.TEST][self.dataset.label_feature])

    def validation_iterator(self) -> Generator[tuple[list[str], ListOfLabels, list[str], ListOfLabels]]:
        """Yield folds for cross-validation."""
        if self.config.scheme != "cv":
            msg = f"Cannot call cross-validation on {self.config.scheme} DataHandler"
            raise RuntimeError(msg)

        for j in range(self.config.n_folds):
            val_utterances = self.train_utterances(j)
            val_labels = self.train_labels(j)
            train_folds = [i for i in range(self.config.n_folds) if i != j]
            train_utterances = [ut for i_fold in train_folds for ut in self.train_utterances(i_fold)]
            train_labels = [lab for i_fold in train_folds for lab in self.train_labels(i_fold)]

            # filter out all OOS samples from train
            train_utterances = [ut for ut, lab in zip(train_utterances, train_labels, strict=True) if lab is not None]
            train_labels = [lab for lab in train_labels if lab is not None]
            yield train_utterances, train_labels, val_utterances, val_labels  # type: ignore[misc]

    def _split_ho(
        self,
        separation_ratio: FloatFromZeroToOne | None,
        validation_size: FloatFromZeroToOne,
        is_few_shot: bool,
        examples_per_intent: int,
    ) -> None:
        has_validation_split = any(split.startswith(Split.VALIDATION) for split in self.dataset)

        if separation_ratio is not None and Split.TRAIN in self.dataset:
            self._split_train(separation_ratio)

        if not has_validation_split:
            self._split_validation_from_train(validation_size, is_few_shot, examples_per_intent)
        elif is_few_shot:
            self._split_few_shot(examples_per_intent)

        for split in self.dataset:
            n_classes_in_split = self.dataset.get_n_classes(split)
            if n_classes_in_split != self._n_classes:
                message = (
                    f"{n_classes_in_split=} for '{split=}' doesn't match initial number of classes ({self._n_classes})"
                )
                raise ValueError(message)

    def _split_few_shot(self, examples_per_intent: int) -> None:
        if Split.TRAIN in self.dataset:
            self.dataset[Split.TRAIN], self.dataset[Split.VALIDATION] = create_few_shot_split(
                self.dataset[Split.TRAIN],
                self.dataset[Split.VALIDATION],
                multilabel=self.dataset.multilabel,
                label_column=self.dataset.label_feature,
                random_seed=self._seed,
                examples_per_label=examples_per_intent,
            )
        else:
            for idx in range(2):
                self.dataset[f"{Split.TRAIN}_{idx}"], self.dataset[f"{Split.VALIDATION}_{idx}"] = create_few_shot_split(
                    self.dataset[f"{Split.TRAIN}_{idx}"],
                    self.dataset[f"{Split.VALIDATION}_{idx}"],
                    multilabel=self.dataset.multilabel,
                    label_column=self.dataset.label_feature,
                    random_seed=self._seed,
                    examples_per_label=examples_per_intent,
                )

    def _split_train(self, ratio: FloatFromZeroToOne) -> None:
        """Split on two sets.

        One is for scoring node optimizaton, one is for decision node.

        Args:
            ratio: Split ratio
        """
        self.dataset[f"{Split.TRAIN}_0"], self.dataset[f"{Split.TRAIN}_1"] = split_dataset(
            self.dataset,
            split=Split.TRAIN,
            test_size=ratio,
            random_seed=self._seed,
            allow_oos_in_train=False,  # only train data for decision node should contain OOS
        )
        self.dataset.pop(Split.TRAIN)

    def _split_cv(self, is_few_shot: bool, examples_per_intent: int) -> None:
        extra_splits = [split_name for split_name in self.dataset if split_name != Split.TEST]
        self.dataset[Split.TRAIN] = concatenate_datasets([self.dataset.pop(split_name) for split_name in extra_splits])

        for j in range(self.config.n_folds - 1):
            self.dataset[Split.TRAIN], self.dataset[f"{Split.TRAIN}_{j}"] = split_dataset(
                self.dataset,
                split=Split.TRAIN,
                test_size=1 / (self.config.n_folds - j),
                random_seed=self._seed,
                is_few_shot=is_few_shot,
                examples_per_intent=examples_per_intent,
                allow_oos_in_train=True,
            )
        self.dataset[f"{Split.TRAIN}_{self.config.n_folds - 1}"] = self.dataset.pop(Split.TRAIN)

    def _split_validation_from_train(self, size: float, is_few_shot: bool, examples_per_intent: int) -> None:
        if Split.TRAIN in self.dataset:
            self.dataset[Split.TRAIN], self.dataset[Split.VALIDATION] = split_dataset(
                self.dataset,
                split=Split.TRAIN,
                test_size=size,
                random_seed=self._seed,
                is_few_shot=is_few_shot,
                examples_per_intent=examples_per_intent,
                allow_oos_in_train=True,
            )
        else:
            for idx in range(2):
                self.dataset[f"{Split.TRAIN}_{idx}"], self.dataset[f"{Split.VALIDATION}_{idx}"] = split_dataset(
                    self.dataset,
                    split=f"{Split.TRAIN}_{idx}",
                    test_size=size,
                    random_seed=self._seed,
                    is_few_shot=is_few_shot,
                    examples_per_intent=examples_per_intent,
                    allow_oos_in_train=idx == 1,  # for decision node it's ok to have oos in train
                )

    def prepare_for_refit(self) -> None:
        """Merge all training folds into one in order to retrain configured optimal pipeline on it."""
        if self.config.scheme == "ho":
            return

        train_folds = [split_name for split_name in self.dataset if split_name.startswith(Split.TRAIN)]
        self.dataset[Split.TRAIN] = concatenate_datasets([self.dataset.pop(name) for name in train_folds])

        self.dataset[f"{Split.TRAIN}_0"], self.dataset[f"{Split.TRAIN}_1"] = split_dataset(
            self.dataset,
            split=Split.TRAIN,
            test_size=self.config.separation_ratio or 0.5,
            random_seed=self._seed,
            allow_oos_in_train=False,
        )

        self.dataset.pop(Split.TRAIN)
