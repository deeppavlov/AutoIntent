"""Prediction metrics for multiclass and multilabel classification tasks."""
import logging
from typing import Protocol

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .converter import transform
from .custom_types import LABELS_VALUE_TYPE

logger = logging.getLogger(__name__)


class PredictionMetricFn(Protocol):
    """Protocol for prediction metrics."""

    def __call__(self, y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
        """
        Calculate prediction metric.

        :param y_true: True values of labels
            - multiclass case: list representing an array shape `(n_samples,)` of integer class labels
            - multilabel case: list representing a matrix of shape `(n_samples, n_classes)` with binary values
        :param y_pred: Predicted values of labels. Same shape as `y_true`
        :return: Score of the prediction metric
        """
        ...


def prediction_accuracy(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    """
    Calculate prediction accuracy.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the prediction accuracy
    """
    y_true_, y_pred_ = transform(y_true, y_pred)
    return np.mean(y_true_ == y_pred_)  # type: ignore[no-any-return]


def _prediction_roc_auc_multiclass(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    """
    Calculate roc_auc for multiclass.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the prediction roc_auc
    """
    y_true_, y_pred_ = transform(y_true, y_pred)

    n_classes = len(np.unique(y_true))
    roc_auc_scores: list[float] = []
    for k in range(n_classes):
        binarized_true = (y_true_ == k).astype(int)
        binarized_pred = (y_pred_ == k).astype(int)
        roc_auc_scores.append(roc_auc_score(binarized_true, binarized_pred))

    return np.mean(roc_auc_scores)  # type: ignore[return-value]


def _prediction_roc_auc_multilabel(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    """
    Calculate roc_auc for multilabel.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the prediction accuracy
    """
    return roc_auc_score(y_true, y_pred, average="macro")  # type: ignore[no-any-return]


def prediction_roc_auc(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    """
    Calculate roc_auc for multiclass and multilabel.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the prediction roc_auc
    """
    y_true_, y_pred_ = transform(y_true, y_pred)
    if y_pred_.ndim == y_true_.ndim == 1:
        return _prediction_roc_auc_multiclass(y_true_, y_pred_)
    if y_pred_.ndim == y_true_.ndim == 2:  # noqa: PLR2004
        return _prediction_roc_auc_multilabel(y_true_, y_pred_)
    msg = "Something went wrong with labels dimensions"
    logger.error(msg)
    raise ValueError(msg)


def prediction_precision(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    """
    Calculate prediction precision.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the prediction precision
    """
    return precision_score(y_true, y_pred, average="macro")  # type: ignore[no-any-return]


def prediction_recall(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    """
    Calculate prediction recall.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the prediction recall
    """
    return recall_score(y_true, y_pred, average="macro")  # type: ignore[no-any-return]


def prediction_f1(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    """
    Calculate prediction f1 score.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the prediction accuracy
    """
    return f1_score(y_true, y_pred, average="macro")  # type: ignore[no-any-return]
