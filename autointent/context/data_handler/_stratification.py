"""Module for dataset splitting.

This module provides utilities for splitting datasets into training and testing sets.
It includes support for both single-label and multi-label stratified splitting.
"""

from collections.abc import Sequence

import numpy as np
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets
from numpy import typing as npt
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification

from autointent import Dataset
from autointent.custom_types import LabelType


class StratifiedSplitter:
    """
    A class for stratified splitting of datasets.

    This class provides methods to split a dataset into training and testing subsets
    while preserving the distribution of target labels. It supports both single-label
    and multi-label datasets.
    """

    def __init__(
        self,
        test_size: float,
        label_feature: str,
        random_seed: int,
        shuffle: bool = True,
    ) -> None:
        """
        Initialize the StratifiedSplitter.

        :param test_size: Proportion of the dataset to include in the test split.
        :param label_feature: Name of the feature containing labels for stratification.
        :param random_seed: Seed for random number generation to ensure reproducibility.
        :param shuffle: Whether to shuffle the data before splitting. Defaults to True.
        """
        self.test_size = test_size
        self.label_feature = label_feature
        self.random_seed = random_seed
        self.shuffle = shuffle

    def __call__(
        self, dataset: HFDataset, multilabel: bool, allow_oos_in_train: bool | None = None
    ) -> tuple[HFDataset, HFDataset]:
        """
        Split the dataset into training and testing subsets.

        :param dataset: The input dataset to be split.
        :param multilabel: Whether the dataset is multi-label.
        :param allow_oos_in_train: Set to True if you want to see out-of-scope utterances in train split.
        :return: A tuple containing the training and testing datasets.
        """
        if not self._has_oos_samples(dataset):
            return self._split_without_oos(dataset, multilabel, self.test_size)
        if allow_oos_in_train is None:
            msg = (
                "Error while splitting dataset. It contains OOS samples, "
                "you need to set the parameter allow_oos_in_train."
            )
            raise ValueError(msg)
        splitter = self._split_allow_oos_in_train if allow_oos_in_train else self._split_disallow_oos_in_train
        return splitter(dataset, multilabel)

    def _has_oos_samples(self, dataset: HFDataset) -> bool:
        oos_samples = dataset.filter(lambda sample: sample[self.label_feature] is None)
        return len(oos_samples) > 0

    def _split_without_oos(self, dataset: HFDataset, multilabel: bool, test_size: float) -> tuple[HFDataset, HFDataset]:
        splitter = self._split_multilabel if multilabel else self._split_multiclass
        splits = splitter(dataset, test_size)
        return dataset.select(splits[0]), dataset.select(splits[1])

    def _split_multiclass(self, dataset: HFDataset, test_size: float) -> Sequence[npt.NDArray[np.int_]]:
        return train_test_split(  # type: ignore[no-any-return]
            np.arange(len(dataset)),
            test_size=test_size,
            random_state=self.random_seed,
            shuffle=self.shuffle,
            stratify=dataset[self.label_feature],
        )

    def _split_multilabel(self, dataset: HFDataset, test_size: float) -> Sequence[npt.NDArray[np.int_]]:
        splitter = IterativeStratification(
            n_splits=2,
            order=2,
            sample_distribution_per_fold=[test_size, 1.0 - test_size],
        )
        return next(splitter.split(np.arange(len(dataset)), np.array(dataset[self.label_feature])))

    def _split_allow_oos_in_train(self, dataset: HFDataset, multilabel: bool) -> tuple[HFDataset, HFDataset]:
        """
        Proportionally distribute OOS samples between two splits.

        Internally, this method creates a dataset copy with some integer assigned as OOS class id.
        With OOS samples treated as a separate class we obtain proportional distribution of them between two splits.
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
        if sample[self.label_feature] == old:
            sample[self.label_feature] = new
        return sample

    def _add_oos_label(self, sample: dict[str, str | LabelType], n_classes: int) -> dict[str, str | LabelType]:
        """Add OOS as a class for multi-label case."""
        if sample[self.label_feature] is None:
            sample[self.label_feature] = [0] * n_classes
        sample[self.label_feature] += [1]  # type: ignore[operator]
        return sample

    def _remove_oos_label(self, sample: dict[str, str | LabelType], n_classes: int) -> dict[str, str | LabelType]:
        """Remove OOS as a class for multi-label case."""
        sample[self.label_feature] = sample[self.label_feature][:-1]  # type: ignore[index]
        if sample[self.label_feature] == [0] * n_classes:
            sample[self.label_feature] = None  # type: ignore[assignment]
        return sample

    def _split_disallow_oos_in_train(self, dataset: HFDataset, multilabel: bool) -> tuple[HFDataset, HFDataset]:
        """
        Move all OOS samples to test split.

        This method preserves the defined test_size proportion so you won't get unexpectedly
        large test set even you have lots of OOS samples.
        """
        in_domain_dataset, out_of_domain_dataset = self._separate_oos(dataset)
        adjusted_test_size = self._get_adjusted_test_size(len(dataset), len(out_of_domain_dataset))
        train, test = self._split_without_oos(in_domain_dataset, multilabel, adjusted_test_size)
        test = concatenate_datasets([test, out_of_domain_dataset])
        return train, test

    def _separate_oos(self, dataset: HFDataset) -> tuple[HFDataset, HFDataset]:
        in_domain_ids = []
        out_of_domain_ids = []
        for i, sample in enumerate(dataset):
            if sample[self.label_feature] is None:
                out_of_domain_ids.append(i)
            else:
                in_domain_ids.append(i)
        return dataset.select(in_domain_ids), dataset.select(out_of_domain_ids)

    def _get_adjusted_test_size(self, n: int, k: int) -> float:
        """
        Calculate effective test_size in order to preserve original proportion.

        :param n: size of original dataset (both with in-domain and out-of-domain samples)
        :param k: number of out-of-domain samples within the dataset
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
    random_seed: int,
    allow_oos_in_train: bool | None = None,
) -> tuple[HFDataset, HFDataset]:
    """
    Split a Dataset object into training and testing subsets.

    This function uses the StratifiedSplitter to perform stratified splitting
    while preserving the distribution of labels.

    :param dataset: The dataset to be split, which must include training data.
    :param split: The specific data split to be divided, e.g., "train" or
        another split within the dataset.
    :param test_size: Proportion of the dataset to include in the test split.
        Should be a float value between 0.0 and 1.0, where 0.0
        means no data will be assigned to the test set, and 1.0
        means all data will be assigned to the test set. For example,
        a value of 0.2 indicates 20% of the data will be used for testing.
    :param random_seed: Seed for random number generation to ensure reproducibility.
    :return: A tuple containing two subsets of the selected split.
    """
    splitter = StratifiedSplitter(
        test_size=test_size,
        label_feature=dataset.label_feature,
        random_seed=random_seed,
    )
    return splitter(dataset[split], dataset.multilabel, allow_oos_in_train=allow_oos_in_train)
