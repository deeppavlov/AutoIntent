import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


class PredictionMetricFn(Protocol):
    def __call__(self, y_true: list[int] | list[list[int]], y_pred: list[int] | list[list[int]]) -> float:
        """
        Arguments
        ---
        - `y_true`:
            - multiclass case: list representing an array shape `(n_samples,)` of integer class labels
            - multilabel case: list representing a matrix of shape `(n_samples, n_classes)` with binary values
        - `y_pred`: same as `y_true`
        """
        ...


def simple_check(func: Callable[[npt.NDArray[Any], npt.NDArray[Any]], float]) -> PredictionMetricFn:
    @wraps(func)
    def wrapper(y_true: list[int] | list[list[int]], y_pred: list[int] | list[list[int]]) -> float:
        y_pred_ = np.array(y_pred)
        y_true_ = np.array(y_true)
        if y_pred_.ndim != y_true_.ndim:
            msg = "Something went wrong with labels dimensions"
            logger.error(msg)
            raise ValueError(msg)
        return func(y_true_, y_pred_)

    return wrapper


@simple_check
def prediction_accuracy(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
    """supports multiclass and multilabel"""
    return np.mean(y_true == y_pred)  # type: ignore[no-any-return]


def _prediction_roc_auc_multiclass(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
    """supports multiclass"""
    n_classes = len(np.unique(y_true))
    roc_auc_scores: list[float] = []
    for k in range(n_classes):
        binarized_true = (y_true == k).astype(int)
        binarized_pred = (y_pred == k).astype(int)
        roc_auc_scores.append(roc_auc_score(binarized_true, binarized_pred))

    return np.mean(roc_auc_scores)


def _prediction_roc_auc_multilabel(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
    """supports multilabel"""
    return roc_auc_score(y_true, y_pred, average="macro")


@simple_check
def prediction_roc_auc(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
    """supports multiclass and multilabel"""
    if y_pred.ndim == y_true.ndim == 1:
        return _prediction_roc_auc_multiclass(y_true, y_pred)
    if y_pred.ndim == y_true.ndim == 2:  # noqa: PLR2004
        return _prediction_roc_auc_multilabel(y_true, y_pred)
    msg = "Something went wrong with labels dimensions"
    logger.error(msg)
    raise ValueError(msg)


@simple_check
def prediction_precision(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
    """supports multiclass and multilabel"""
    return precision_score(y_true, y_pred, average="macro")


@simple_check
def prediction_recall(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
    """supports multiclass and multilabel"""
    return recall_score(y_true, y_pred, average="macro")


@simple_check
def prediction_f1(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
    """supports multiclass and multilabel"""
    return f1_score(y_true, y_pred, average="macro")
