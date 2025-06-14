"""Module for dataset splitting.

This module provides utilities for splitting datasets into training and testing sets.
It includes support for both single-label and multi-label stratified splitting.
"""

import logging
from collections.abc import Sequence

import numpy as np
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets
from numpy import typing as npt
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification
from transformers import set_seed  # type: ignore[attr-defined]

from autointent import Dataset
from autointent.custom_types import LabelType

logger = logging.getLogger(__name__)


class StratifiedSplitter:
    """A class for stratified splitting of datasets.

    This class provides methods to split a dataset into training and testing subsets
    while preserving the distribution of target labels. It supports both single-label
    and multi-label datasets.
    """

    def __init__(
        self,
        test_size: float,
        label_feature: str,
        random_seed: int | None,
        shuffle: bool = True,
        is_few_shot: bool = False,
        examples_per_label: int = 8,
    ) -> None:
        """Initialize the StratifiedSplitter.

        Args:
            test_size: Proportion of the dataset to include in the test split.
            label_feature: Name of the feature containing labels for stratification.
            random_seed: Seed for random number generation to ensure reproducibility.
            shuffle: Whether to shuffle the data before splitting.
            is_few_shot: Whether the dataset is a few-shot dataset.
            examples_per_label: Number of examples per label for few-shot datasets.
        """
        self.test_size = test_size
        self.label_feature = label_feature
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.is_few_shot = is_few_shot
        self.examples_per_label = examples_per_label

    def __call__(
        self, dataset: HFDataset, multilabel: bool, allow_oos_in_train: bool | None = None
    ) -> tuple[HFDataset, HFDataset]:
        """Split the dataset into training and testing subsets.

        Args:
            dataset: The input dataset to be split.
            multilabel: Whether the dataset is multi-label.
            allow_oos_in_train: Set to True if you want to see out-of-scope utterances in train split.

        Returns:
            A tuple containing the training and testing datasets.

        Raises:
            ValueError: If OOS samples are present but allow_oos_in_train is not specified.
        """
        if not self._has_oos_samples(dataset):
            train, test = self._split_without_oos(dataset, multilabel, self.test_size)
            if self.is_few_shot:
                train, test = create_few_shot_split(
                    train,
                    test,
                    multilabel=multilabel,
                    label_column=self.label_feature,
                    examples_per_label=self.examples_per_label,
                )
            return train, test
        if allow_oos_in_train is None:
            msg = (
                "Error while splitting dataset. It contains OOS samples, "
                "you need to set the parameter allow_oos_in_train."
            )
            raise ValueError(msg)
        splitter = self._split_allow_oos_in_train if allow_oos_in_train else self._split_disallow_oos_in_train
        train, test = splitter(dataset, multilabel)
        if self.is_few_shot:
            train, test = create_few_shot_split(
                train,
                test,
                multilabel=multilabel,
                label_column=self.label_feature,
                examples_per_label=self.examples_per_label,
            )
        return train, test

    def _has_oos_samples(self, dataset: HFDataset) -> bool:
        """Check if the dataset contains out-of-scope samples.

        Args:
            dataset: The dataset to check.

        Returns:
            True if the dataset contains OOS samples, False otherwise.
        """
        oos_samples = dataset.filter(lambda sample: sample[self.label_feature] is None)
        return len(oos_samples) > 0

    def _split_without_oos(self, dataset: HFDataset, multilabel: bool, test_size: float) -> tuple[HFDataset, HFDataset]:
        """Split dataset that doesn't contain OOS samples.

        Args:
            dataset: Dataset to split.
            multilabel: Whether the dataset is multi-label.
            test_size: Proportion of the dataset to include in the test split.

        Returns:
            A tuple containing training and testing datasets.
        """
        splitter = self._split_multilabel if multilabel else self._split_multiclass
        splits = splitter(dataset, test_size)
        return dataset.select(splits[0]), dataset.select(splits[1])

    def _split_multiclass(self, dataset: HFDataset, test_size: float) -> Sequence[npt.NDArray[np.int_]]:
        """Split multiclass dataset.

        Args:
            dataset: Dataset to split.
            test_size: Proportion of the dataset to include in the test split.

        Returns:
            A sequence containing indices for train and test splits.
        """
        return train_test_split(  # type: ignore[no-any-return]
            np.arange(len(dataset)),
            test_size=test_size,
            random_state=self.random_seed,
            shuffle=self.shuffle,
            stratify=dataset[self.label_feature],
        )

    def _split_multilabel(self, dataset: HFDataset, test_size: float) -> Sequence[npt.NDArray[np.int_]]:
        """Split multilabel dataset.

        Args:
            dataset: Dataset to split.
            test_size: Proportion of the dataset to include in the test split.

        Returns:
            A sequence containing indices for train and test splits.
        """
        if self.random_seed is not None:
            set_seed(self.random_seed)  # workaround for buggy nature of IterativeStratification from skmultilearn
        splitter = IterativeStratification(
            n_splits=2,
            order=2,
            sample_distribution_per_fold=[test_size, 1.0 - test_size],
        )
        return next(splitter.split(np.arange(len(dataset)), np.array(dataset[self.label_feature])))

    def _split_allow_oos_in_train(self, dataset: HFDataset, multilabel: bool) -> tuple[HFDataset, HFDataset]:
        """Proportionally distribute OOS samples between two splits.

        Internally creates a dataset copy with some integer assigned as OOS class id.
        With OOS samples treated as a separate class we obtain proportional distribution
        of them between two splits.

        Args:
            dataset: Dataset to split.
            multilabel: Whether the dataset is multi-label.

        Returns:
            A tuple containing training and testing datasets.
        """
        # add oos as a class
        if multilabel:
            in_domain_sample = next(sample for sample in dataset if sample[self.label_feature] is not None)
            n_classes = len(in_domain_sample[self.label_feature])
            dataset = dataset.map(self._add_oos_label, fn_kwargs={"n_classes": n_classes})
        else:
            oos_class_id = len(dataset.unique(self.label_feature)) - 1
            dataset = dataset.map(self._map_label, fn_kwargs={"old": None, "new": oos_class_id})

        # perform stratified splitting
        train, test = self._split_without_oos(dataset, multilabel=False, test_size=self.test_size)

        # remove oos as a class
        if multilabel:
            train = train.map(self._remove_oos_label, fn_kwargs={"n_classes": n_classes})
            test = test.map(self._remove_oos_label, fn_kwargs={"n_classes": n_classes})
        else:
            train = train.map(self._map_label, fn_kwargs={"old": oos_class_id, "new": None})
            test = test.map(self._map_label, fn_kwargs={"old": oos_class_id, "new": None})

        return train, test

    def _map_label(
        self, sample: dict[str, str | LabelType], old: LabelType, new: LabelType
    ) -> dict[str, str | LabelType]:
        """Map labels from old value to new value.

        Args:
            sample: Sample containing the label to map.
            old: Old label value.
            new: New label value.

        Returns:
            Sample with mapped label.
        """
        if sample[self.label_feature] == old:
            sample[self.label_feature] = new
        return sample

    def _add_oos_label(self, sample: dict[str, str | LabelType], n_classes: int) -> dict[str, str | LabelType]:
        """Add OOS as a class for multi-label case.

        Args:
            sample: Sample to modify.
            n_classes: Number of classes in the dataset.

        Returns:
            Sample with added OOS label.
        """
        if sample[self.label_feature] is None:
            sample[self.label_feature] = [0] * n_classes
        sample[self.label_feature] += [1]  # type: ignore[operator]
        return sample

    def _remove_oos_label(self, sample: dict[str, str | LabelType], n_classes: int) -> dict[str, str | LabelType]:
        """Remove OOS as a class for multi-label case.

        Args:
            sample: Sample to modify.
            n_classes: Number of classes in the dataset.

        Returns:
            Sample with removed OOS label.
        """
        sample[self.label_feature] = sample[self.label_feature][:-1]  # type: ignore[index]
        if sample[self.label_feature] == [0] * n_classes:
            sample[self.label_feature] = None  # type: ignore[assignment]
        return sample

    def _split_disallow_oos_in_train(self, dataset: HFDataset, multilabel: bool) -> tuple[HFDataset, HFDataset]:
        """Move all OOS samples to test split.

        This method preserves the defined test_size proportion so you won't get unexpectedly
        large test set even you have lots of OOS samples.

        Args:
            dataset: Dataset to split.
            multilabel: Whether the dataset is multi-label.

        Returns:
            A tuple containing training and testing datasets.
        """
        in_domain_dataset, out_of_domain_dataset = self._separate_oos(dataset)
        adjusted_test_size = self._get_adjusted_test_size(len(dataset), len(out_of_domain_dataset))
        train, test = self._split_without_oos(in_domain_dataset, multilabel, adjusted_test_size)
        test = concatenate_datasets([test, out_of_domain_dataset])
        return train, test

    def _separate_oos(self, dataset: HFDataset) -> tuple[HFDataset, HFDataset]:
        """Separate OOS samples from in-domain samples.

        Args:
            dataset: Dataset to separate.

        Returns:
            A tuple containing in-domain and out-of-domain datasets.
        """
        in_domain_ids = []
        out_of_domain_ids = []
        for i, sample in enumerate(dataset):
            if sample[self.label_feature] is None:
                out_of_domain_ids.append(i)
            else:
                in_domain_ids.append(i)
        return dataset.select(in_domain_ids), dataset.select(out_of_domain_ids)

    def _get_adjusted_test_size(self, n: int, k: int) -> float:
        """Calculate effective test_size to preserve original proportion.

        Args:
            n: Size of original dataset (both with in-domain and out-of-domain samples).
            k: Number of out-of-domain samples within the dataset.

        Returns:
            Adjusted test size.

        Raises:
            ValueError: If dataset contains too many OOS samples.
        """
        if k == 0:
            return self.test_size
        res = (self.test_size * n - k) / (n - k)
        if res <= 0:
            msg = (
                "Error while splitting dataset. Dataset contains too many OOS samples. "
                "Either increase test_size or decrease number of OOS samples."
            )
            raise ValueError(msg)
        return res


def split_dataset(
    dataset: Dataset,
    split: str,
    test_size: float,
    random_seed: int | None,
    is_few_shot: bool = False,
    examples_per_intent: int = 8,
    allow_oos_in_train: bool | None = None,
) -> tuple[HFDataset, HFDataset]:
    """Split a Dataset object into training and testing subsets.

    Args:
        dataset: The dataset to split, which must include training data.
        split: The specific data split to divide.
        test_size: Proportion of the dataset to include in the test split.
        random_seed: Seed for random number generation.
        is_few_shot: Whether the dataset is a few-shot dataset.
        examples_per_intent: Number of examples per label for few-shot datasets.
        allow_oos_in_train: Whether to allow OOS samples in train split.

    Returns:
        A tuple containing two subsets of the selected split.
    """
    splitter = StratifiedSplitter(
        test_size=test_size,
        label_feature=dataset.label_feature,
        random_seed=random_seed,
        is_few_shot=is_few_shot,
        examples_per_label=examples_per_intent,
    )
    return splitter(dataset[split], dataset.multilabel, allow_oos_in_train=allow_oos_in_train)


def create_few_shot_split(
    train_dataset: HFDataset,
    validation_dataset: HFDataset,
    label_column: str,
    examples_per_label: int = 8,
    multilabel: bool = False,
    random_seed: int | None = None,
) -> tuple[HFDataset, HFDataset]:
    """Create a few-shot dataset split with a specified number of examples per label.

    Args:
        train_dataset: A Hugging Face dataset or DatasetDict
        validation_dataset: A Hugging Face dataset or DatasetDict
        label_column: The name of the column containing labels (default: 'label')
        examples_per_label: Number of examples to include per label in the train split (default: 8)
        multilabel: Whether the dataset is multi-label (default: False)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        A tuple containing the train and validation datasets.
    """
    # Add a unique index column to track examples
    train_dataset = train_dataset.add_column("__index__", list(range(len(train_dataset))))
    if multilabel:
        _unique_labels = set()
        for example in train_dataset:
            if example[label_column] is not None:
                _unique_labels.add(tuple(example[label_column]))
        unique_labels = list(_unique_labels)
    else:
        unique_labels = train_dataset.unique(label_column)

    # Create train dataset by sampling examples_per_label for each label
    train_datasets = []
    selected_indices = set()

    for label in unique_labels:
        if multilabel:
            label_examples = train_dataset.filter(lambda row: tuple(row[label_column]) == label)  # noqa: B023
        else:
            label_examples = train_dataset.filter(lambda row: row[label_column] == label)  # noqa: B023
        label_examples = label_examples.shuffle(seed=random_seed)

        num_to_select = min(examples_per_label, len(label_examples))
        selected_examples = label_examples.select(range(num_to_select))

        if num_to_select < examples_per_label:
            msg = (
                f"Warning: Only {num_to_select} examples available for label '{label}', "
                f"which is less than the requested {examples_per_label}"
            )
            logger.warning(msg)

        train_datasets.append(selected_examples)
        selected_indices.update([ex["__index__"] for ex in selected_examples])

    # Create validation split with remaining examples
    extra_validation_dataset = train_dataset.filter(
        lambda example: example["__index__"] not in selected_indices
    ).remove_columns("__index__")

    validation_dataset = concatenate_datasets([validation_dataset, extra_validation_dataset])
    train_dataset = concatenate_datasets(train_datasets).remove_columns("__index__")

    return train_dataset, validation_dataset
