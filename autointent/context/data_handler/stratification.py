import itertools as it
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from sklearn.utils import _safe_indexing, indexable
from skmultilearn.model_selection import IterativeStratification

from .schemas import Dataset, DatasetType


def get_sample_utterances(intent_records: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    """get plain list of all sample utterances and their intent labels"""
    utterances = [intent["sample_utterances"] for intent in intent_records]
    labels = [[intent["intent_id"]] * len(uts) for intent, uts in zip(intent_records, utterances, strict=False)]

    utterances = list(it.chain.from_iterable(utterances))
    labels = list(it.chain.from_iterable(labels))

    return utterances, labels


def get_samples(dataset: Dataset) -> tuple[list[str], list[int | list[int] | None]]:
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
    return [utterance.text for utterance in dataset.utterances if utterance.oos]


def split_sample_utterances(
    dataset: Dataset,
    test_dataset: Dataset | None = None,
    random_seed: int = 0,
) -> tuple[
    int,
    list[Any],
    list[str],
    list[str],
    list[int],
    list[int],
]:
    """
    Return: n_classes, oos_utterances, utterances_train, utterances_test, labels_train, labels_test
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
        msg = "for some reason test set doesn't contain some classes examples"
        logger.error(msg)
        raise ValueError(msg)

    return dataset.n_classes, oos_utterances, *splits


def multilabel_train_test_split(
    *arrays: npt.NDArray[Any],
    test_size: float = 0.25,
    random_state: int = 0,
    shuffle: bool = False,
    stratify: list[list[int]] | None = None,
) -> list[npt.NDArray[Any]]:
    """
    TODO:
    - test whether this function is not random
    - implement shuffling
    """
    if stratify is None:
        return train_test_split(*arrays, test_size=test_size, random_state=random_state, shuffle=shuffle)  # type: ignore[no-any-return]
    n_arrays = len(arrays)
    if n_arrays == 0:
        msg = "At least one array required as input"
        raise ValueError(msg)

    arrays = indexable(*arrays)

    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0 - test_size])
    train, test = next(stratifier.split(arrays[0], np.array(stratify)))

    return list(it.chain.from_iterable((_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays))


def validate_test_labels(test_labels: list[int] | list[list[int]], multilabel: bool, n_classes: int) -> bool:
    """
    ensure that all classes are presented in the presented labels set
    """
    if not multilabel and isinstance(test_labels[0], int):
        return is_multiclass_test_set_complete(test_labels, n_classes)  # type: ignore[arg-type]
    if multilabel and isinstance(test_labels[0], list):
        return is_multilabel_test_set_complete(np.array(test_labels))
    msg = "unexpected labels format"
    raise ValueError(msg)


def is_multilabel_test_set_complete(labels: npt.NDArray[Any]) -> bool:
    labels_counts = labels.sum(axis=0)
    return (labels_counts > 0).all()  # type: ignore[no-any-return]


def is_multiclass_test_set_complete(labels: list[int], n_classes: int) -> bool:
    ohe_labels = to_onehot(np.array(labels), n_classes)
    return is_multilabel_test_set_complete(ohe_labels)


def to_onehot(labels: npt.NDArray[Any], n_classes: int) -> npt.NDArray[Any]:
    """convert nd array of ints to (n+1)d array of zeros and ones"""
    new_shape = (*labels.shape, n_classes)
    onehot_labels = np.zeros(shape=new_shape)
    indices = (*tuple(np.indices(labels.shape)), labels)
    onehot_labels[indices] = 1
    return onehot_labels
