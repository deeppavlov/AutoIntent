"""Module for dataset processing and splitting.

This module provides utilities for handling dataset samples, splitting datasets
into training and testing sets, and validating test sets to ensure class coverage.
"""

import itertools as it
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from sklearn.utils import _safe_indexing, indexable
from skmultilearn.model_selection import IterativeStratification

from autointent.custom_types import LabelType

from ._schemas import Dataset, DatasetType


def get_sample_utterances(intent_records: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    """
    Get plain lists of sample utterances and their intent labels.

    :param intent_records: List of intent records, each containing sample utterances and intent IDs.
    :return: A tuple of two lists:
             - utterances: List of all sample utterances.
             - labels: List of corresponding intent labels.
    """
    utterances = [intent["sample_utterances"] for intent in intent_records]
    labels = [[intent["intent_id"]] * len(uts) for intent, uts in zip(intent_records, utterances, strict=False)]

    utterances = list(it.chain.from_iterable(utterances))
    labels = list(it.chain.from_iterable(labels))

    return utterances, labels


def get_samples(dataset: Dataset) -> tuple[list[str], list[LabelType | None]]:
    """
    Extract samples (utterances and labels) from the dataset.

    :param dataset: A `Dataset` object containing utterances and their labels.
    :return: A tuple containing:
             - utterances: List of utterance texts.
             - labels: List of labels (either one-hot encoded or original).
    """
    utterances, labels = [], []
    for utterance in dataset.utterances:
        if utterance.oos:
            continue
        utterances.append(utterance.text)
        if dataset.type == DatasetType.multiclass:
            labels.append(utterance.label)
        else:
            labels.append(utterance.one_hot_label(dataset.n_classes))
    return utterances, labels


def get_oos_samples(dataset: Dataset) -> list[str]:
    """
    Extract Out-of-Scope (OOS) samples from the dataset.

    :param dataset: A `Dataset` object.
    :return: A list of OOS utterance texts.
    """
    return [utterance.text for utterance in dataset.utterances if utterance.oos]


def split_sample_utterances(
    dataset: Dataset,
    test_dataset: Dataset | None = None,
    random_seed: int = 0,
) -> tuple[
    int,
    list[str],
    list[str],
    list[str],
    list[LabelType],
    list[LabelType],
]:
    """
    Split dataset into training and testing sets.

    :param dataset: A `Dataset` object containing utterances and labels.
    :param test_dataset: Optional `Dataset` object for testing. If provided, it will be used as the test set.
    :param random_seed: Random seed for reproducibility.
    :return: A tuple containing:
             - n_classes: Number of unique classes.
             - oos_utterances: List of OOS utterance texts.
             - utterances_train: Training utterance texts.
             - utterances_test: Testing utterance texts.
             - labels_train: Training labels.
             - labels_test: Testing labels.
    """
    logger = logging.getLogger(__name__)

    utterances, labels = get_samples(dataset)
    oos_utterances = get_oos_samples(dataset)

    splitter = train_test_split
    if dataset.type == DatasetType.multilabel:
        splitter = multilabel_train_test_split

    if test_dataset is None:
        splits = splitter(
            utterances,
            labels,
            test_size=0.25,
            random_state=random_seed,
            stratify=labels,
            shuffle=True,
        )
        test_labels = splits[-1]
    else:
        test_utterances, test_labels = get_samples(test_dataset)
        oos_utterances.extend(get_oos_samples(test_dataset))
        splits = [utterances, test_utterances, labels, test_labels]

    is_valid = validate_test_labels(
        test_labels,
        dataset.type == DatasetType.multilabel,
        dataset.n_classes,
    )
    if not is_valid:
        msg = "The test set does not contain examples for some classes."
        logger.error(msg)
        raise ValueError(msg)

    return dataset.n_classes, oos_utterances, *splits


def multilabel_train_test_split(
    *arrays: npt.NDArray[Any],
    test_size: float = 0.25,
    random_state: int = 0,
    shuffle: bool = False,
    stratify: list[LabelType] | None = None,
) -> list[npt.NDArray[Any]]:
    """
    Perform a train-test split for multilabel data.

    :param arrays: Arrays to split, such as features and labels.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random seed for reproducibility.
    :param shuffle: Whether to shuffle the data (not implemented).
    :param stratify: List of multilabel binary format labels for stratification.
    :return: A list of arrays split into training and testing sets.
    :raises ValueError: If no input arrays are provided.
    """
    if stratify is None:
        return train_test_split(*arrays, test_size=test_size, random_state=random_state, shuffle=shuffle)  # type: ignore[no-any-return]
    n_arrays = len(arrays)
    if n_arrays == 0:
        msg = "At least one array is required as input."
        raise ValueError(msg)

    arrays = indexable(*arrays)
    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0 - test_size])
    train, test = next(stratifier.split(arrays[0], np.array(stratify)))

    return list(it.chain.from_iterable((_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays))


def validate_test_labels(test_labels: list[LabelType], multilabel: bool, n_classes: int) -> bool:
    """
    Validate that all classes are represented in the test set.

    :param test_labels: List of test labels.
    :param multilabel: Whether the dataset is multilabel.
    :param n_classes: Total number of classes.
    :return: True if all classes are represented, False otherwise.
    :raises ValueError: If the label format is unexpected.
    """
    if not multilabel and isinstance(test_labels[0], int):
        return is_multiclass_test_set_complete(test_labels, n_classes)  # type: ignore[arg-type]
    if multilabel and isinstance(test_labels[0], list):
        return is_multilabel_test_set_complete(np.array(test_labels))
    msg = "Unexpected labels format."
    raise ValueError(msg)


def is_multilabel_test_set_complete(labels: npt.NDArray[Any]) -> bool:
    """
    Check if all classes are represented in a multilabel test set.

    :param labels: Multilabel test set represented as a binary matrix.
    :return: True if all classes are represented, False otherwise.
    """
    labels_counts = labels.sum(axis=0)
    return (labels_counts > 0).all()  # type: ignore[no-any-return]


def is_multiclass_test_set_complete(labels: list[int], n_classes: int) -> bool:
    """
    Check if all classes are represented in a multiclass test set.

    :param labels: List of multiclass labels.
    :param n_classes: Total number of classes.
    :return: True if all classes are represented, False otherwise.
    """
    ohe_labels = to_onehot(np.array(labels), n_classes)
    return is_multilabel_test_set_complete(ohe_labels)


def to_onehot(labels: npt.NDArray[Any], n_classes: int) -> npt.NDArray[Any]:
    """
    Convert an array of integer labels to a one-hot encoded array.

    :param labels: Array of integer labels.
    :param n_classes: Total number of classes.
    :return: A one-hot encoded array.
    """
    new_shape = (*labels.shape, n_classes)
    onehot_labels = np.zeros(shape=new_shape)
    indices = (*tuple(np.indices(labels.shape)), labels)
    onehot_labels[indices] = 1
    return onehot_labels
