from typing import Protocol

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


class PredictionMetricFn(Protocol):
    def __call__(self, y_true: list[int] | list[list[int]], y_pred: list[int] | list[list[int]]) -> float: ...


def prediction_accuracy(y_true: list[int] | list[list[int]], y_pred: list[int] | list[list[int]]):
    """supports multiclass and multilabel"""
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    assert y_pred.ndim == y_true.ndim
    return np.mean(y_true == y_pred)


def _prediction_roc_auc_multiclass(y_true: list[int], y_pred: list[int]):
    """supports multiclass"""
    n_classes = len(np.unique(y_true))
    roc_auc_scores = []
    for k in range(n_classes):
        binarized_true = (y_true == k).astype(int)
        binarized_pred = (y_pred == k).astype(int)
        roc_auc = roc_auc_score(binarized_true, binarized_pred)
        roc_auc_scores.append(roc_auc)

    macro_roc_auc = np.mean(roc_auc_scores)

    return macro_roc_auc


def _prediction_roc_auc_multilabel(y_true: list[list[int]], y_pred: list[list[int]]):
    """supports multilabel"""
    return roc_auc_score(y_true, y_pred, average="macro")


def prediction_roc_auc(y_true: list[int] | list[list[int]], y_pred: list[int] | list[list[int]]):
    """supports multiclass and multilabel"""
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    if y_pred.ndim == y_true.ndim == 1:
        return _prediction_roc_auc_multiclass(y_true, y_pred)
    if y_pred.ndim == y_true.ndim == 2:
        return _prediction_roc_auc_multilabel(y_true, y_pred)
    raise ValueError("shapes mismatch")


def prediction_precision(y_true: list[int] | list[list[int]], y_pred: list[int] | list[list[int]]):
    """supports multiclass and multilabel"""
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    assert y_pred.ndim == y_true.ndim
    return precision_score(y_true, y_pred, average="macro")


def prediction_recall(y_true: list[int] | list[list[int]], y_pred: list[int] | list[list[int]]):
    """supports multiclass and multilabel"""
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    assert y_pred.ndim == y_true.ndim
    return recall_score(y_true, y_pred, average="macro")


def prediction_f1(y_true: list[int] | list[list[int]], y_pred: list[int] | list[list[int]]):
    """supports multiclass and multilabel"""
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    assert y_pred.ndim == y_true.ndim
    return f1_score(y_true, y_pred, average="macro")
