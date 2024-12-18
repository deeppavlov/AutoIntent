"""Module for dataset splitting.

This module provides utilities for splitting datasets into training and testing sets.
It includes support for both single-label and multi-label stratified splitting.
"""

from collections.abc import Sequence

import numpy as np
from datasets import Dataset as HFDataset
from numpy import typing as npt
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification

from autointent import Dataset


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

    def __call__(self, dataset: HFDataset, multilabel: bool) -> tuple[HFDataset, HFDataset]:
        """
        Split the dataset into training and testing subsets.

        :param dataset: The input dataset to be split.
        :param multilabel: Whether the dataset is multi-label.
        :return: A tuple containing the training and testing datasets.
        """
        splits = self._split_multilabel(dataset) if multilabel else self._split(dataset)
        return dataset.select(splits[0]), dataset.select(splits[1])

    def _split(self, dataset: HFDataset) -> Sequence[npt.NDArray[np.int_]]:
        return train_test_split(  # type: ignore[no-any-return]
            np.arange(len(dataset)),
            test_size=self.test_size,
            random_state=self.random_seed,
            shuffle=self.shuffle,
            stratify=dataset[self.label_feature],
        )

    def _split_multilabel(self, dataset: HFDataset) -> Sequence[npt.NDArray[np.int_]]:
        splitter = IterativeStratification(
            n_splits=2,
            order=2,
            sample_distribution_per_fold=[self.test_size, 1.0 - self.test_size],
        )
        return next(splitter.split(np.arange(len(dataset)), np.array(dataset[self.label_feature])))


def split_dataset(
    dataset: Dataset,
    split: str,
    test_size: float,
    random_seed: int,
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
    return splitter(dataset[split], dataset.multilabel)
