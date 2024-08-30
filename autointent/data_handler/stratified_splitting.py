import itertools as it

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import _safe_indexing, indexable
from skmultilearn.model_selection import IterativeStratification


def get_sample_utterances(intent_records: list[dict]):
    """get plain list of all sample utterances and their intent labels"""
    utterances = [intent["sample_utterances"] for intent in intent_records]
    labels = [
        [intent["intent_id"]] * len(uts) for intent, uts in zip(intent_records, utterances)
    ]

    utterances = list(it.chain.from_iterable(utterances))
    labels = list(it.chain.from_iterable(labels))

    return utterances, labels


def split_sample_utterances(intent_records: list[dict], multilabel: bool):
    """
    Return: utterances_train, utterances_test, labels_train, labels_test

    TODO: ensure stratified train test splitting (test set must contain all classes)
    """

    if not multilabel:
        utterances, labels = get_sample_utterances(intent_records)
        in_domain_mask = np.array(labels) != -1

        in_domain_utterances = [ut for ut, is_in_domain in zip(utterances, in_domain_mask) if is_in_domain]
        in_domain_labels = [lab for lab, is_in_domain in zip(labels, in_domain_mask) if is_in_domain]
        oos_utterances = [ut for ut, is_in_domain in zip(utterances, in_domain_mask) if not is_in_domain]
        
        n_classes = len(set(in_domain_labels))
        splits = train_test_split(
            in_domain_utterances,
            in_domain_labels,
            test_size=0.25,
            random_state=0,
            stratify=in_domain_labels,
            shuffle=True,
        )
    else:
        utterance_records = intent_records
        utterances = [dct["utterance"] for dct in utterance_records]
        labels = [dct["labels"] for dct in utterance_records]

        n_classes = len(set(it.chain.from_iterable(labels)))

        in_domain_utterances = [ut for ut, lab in zip(utterances, labels) if len(lab) > 0]
        in_domain_labels = [[int(i in lab) for i in range(n_classes)] for lab in labels if len(lab) > 0]    # binary labels
        oos_utterances = [ut for ut, lab in zip(utterances, labels) if len(lab) == 0]
        
        splits = multilabel_train_test_split(
            in_domain_utterances,
            in_domain_labels,
            test_size=0.25,
        )
    
    res = [n_classes, oos_utterances] + splits
    return res


def multiclass_labels_as_metadata(labels_list: list[int]):
    return [{"intent_id": lab} for lab in labels_list]


def multilabel_labels_as_metadata(labels_list: list[list[int]], n_classes):
    """labels_list is already in binary format"""
    return [{str(i): lab for i, lab in enumerate(labs)} for labs in labels_list]


def multiclass_metadata_as_labels(metadata: list[dict]):
    return [dct["intent_id"] for dct in metadata]


def multilabel_metadata_as_labels(metadata: list[dict], n_classes):
    return [[dct[str(i)] for i in range(n_classes)] for dct in metadata]


def multilabel_train_test_split(*arrays, stratify=None, test_size=0.25):
    if stratify is None:
        stratify = np.array(arrays[-1])

    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)

    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0-test_size])
    train, test = next(stratifier.split(arrays[0], stratify))

    return list(
        it.chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )