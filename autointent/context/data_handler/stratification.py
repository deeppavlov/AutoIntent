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


def split_sample_utterances(
    dataset: Dataset,
    test_dataset: Dataset | None = None,
    random_seed: int = 0,
) -> ...:
    """
    Return: n_classes, oos_utterances, utterances_train, utterances_test, labels_train, labels_test
    """
    logger = logging.getLogger(__name__)

    multilabel = dataset.type == DatasetType.multilabel

    utterances = [utterance.text for utterance in dataset.utterances]
    labels = [utterance.label for utterance in dataset.utterances]

    if not multilabel:
        logger.debug("parsing multiclass dataset...")

        in_domain_mask = np.array(labels) != -1

        in_domain_utterances = [
            ut for ut, is_in_domain in zip(utterances, in_domain_mask, strict=False) if is_in_domain
        ]
        in_domain_labels = [lab for lab, is_in_domain in zip(labels, in_domain_mask, strict=False) if is_in_domain]
        oos_utterances = [ut for ut, is_in_domain in zip(utterances, in_domain_mask, strict=False) if not is_in_domain]

        n_classes = len(set(in_domain_labels))
        splitter = train_test_split

    else:
        logger.debug("parsing multilabel dataset...")

        classes = set(it.chain.from_iterable(labels))
        n_classes = len(classes) if -1 not in classes else len(classes) - 1

        in_domain_utterances = [ut for ut, lab in zip(utterances, labels, strict=False) if len(lab) > 0]
        in_domain_labels = [[int(i in lab) for i in range(n_classes)] for lab in labels if len(lab) > 0]
        oos_utterances = [ut for ut, lab in zip(utterances, labels, strict=False) if len(lab) == 0]

        splitter = multilabel_train_test_split

    if test_dataset is None:
        logger.debug("test utterances are not provided, using train test splitting...")

        splits = splitter(
            in_domain_utterances,
            in_domain_labels,
            test_size=0.25,
            random_state=random_seed,
            stratify=in_domain_labels,
            shuffle=True,
        )
        test_labels = splits[-1]
    else:
        logger.debug("parsing test dataset...")

        test_utterances, test_labels, oos_utterances = [], [], []

        for utterance in test_dataset.utterances:
            if utterance.oos:
                oos_utterances.append(utterance.text)
            else:
                test_utterances.append(utterance.text)
                if multilabel:
                    test_labels.append(utterance.one_hot_label(n_classes))
                else:
                    test_labels.append(utterance.label)

        splits = [in_domain_utterances, test_utterances, in_domain_labels, test_labels]

    is_valid = validate_test_labels(test_labels, multilabel, n_classes)
    if not is_valid:
        msg = "for some reason test set doesn't contain some classes examples"
        logger.error(msg)
        raise ValueError(msg)

    return n_classes, oos_utterances, *splits


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
        return train_test_split(*arrays, test_size=test_size, random_state=random_state, shuffle=shuffle)
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
        return is_multiclass_test_set_complete(test_labels, n_classes)
    if multilabel and isinstance(test_labels[0], list):
        return is_multilabel_test_set_complete(np.array(test_labels))
    msg = "unexpected labels format"
    raise ValueError(msg)


def is_multilabel_test_set_complete(labels: npt.NDArray[Any]) -> bool:
    labels_counts = labels.sum(axis=0)
    return (labels_counts > 0).all()


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
