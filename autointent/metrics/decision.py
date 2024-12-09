"""Prediction metrics for multiclass and multilabel classification tasks."""

import logging
from typing import Protocol

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from ._converter import transform
from .custom_types import LABELS_VALUE_TYPE

logger = logging.getLogger(__name__)


class DecisionMetricFn(Protocol):
    """Protocol for decision metrics."""

    def __call__(self, y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
        """
        Calculate decision metric.

        :param y_true: True values of labels
            - multiclass case: list representing an array shape `(n_samples,)` of integer class labels
            - multilabel case: list representing a matrix of shape `(n_samples, n_classes)` with binary values
        :param y_pred: Predicted values of labels. Same shape as `y_true`
        :return: Score of the decision metric
        """
        ...


def decision_accuracy(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    r"""
    Calculate decision accuracy. Supports both multiclass and multilabel.

    The decision accuracy is calculated as:

    .. math::

        \text{Accuracy} = \frac{\sum_{i=1}^N \mathbb{1}(y_{\text{true},i} = y_{\text{pred},i})}{N}

    where:
    - :math:`N` is the total number of samples,
    - :math:`y_{\text{true},i}` is the true label for the :math:`i`-th sample,
    - :math:`y_{\text{pred},i}` is the predicted label for the :math:`i`-th sample,
    - :math:`\mathbb{1}(\text{condition})` is the indicator function that equals 1 if the condition
    is true and 0 otherwise.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the decision accuracy
    """
    y_true_, y_pred_ = transform(y_true, y_pred)
    return np.mean(y_true_ == y_pred_)  # type: ignore[no-any-return]


def _decision_roc_auc_multiclass(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    r"""
    Calculate roc_auc for multiclass.

    The ROC AUC score for multiclass is calculated as the mean ROC AUC score
    across all classes, where each class is treated as a binary classification task
    (one-vs-rest).

    .. math::

        \text{ROC AUC}_{\text{multiclass}} = \frac{1}{K} \sum_{k=1}^K \text{ROC AUC}_k

    where:
    - :math:`K` is the number of classes,
    - :math:`\text{ROC AUC}_k` is the ROC AUC score for the :math:`k`-th class,
    calculated by treating it as a binary classification problem (class :math:`k` vs rest).

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the decision roc_auc
    """
    y_true_, y_pred_ = transform(y_true, y_pred)

    n_classes = len(np.unique(y_true))
    roc_auc_scores: list[float] = []
    for k in range(n_classes):
        binarized_true = (y_true_ == k).astype(int)
        binarized_pred = (y_pred_ == k).astype(int)
        roc_auc_scores.append(roc_auc_score(binarized_true, binarized_pred))

    return np.mean(roc_auc_scores)  # type: ignore[return-value]


def _decision_roc_auc_multilabel(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    r"""
    Calculate roc_auc for multilabel.

    This function internally uses :func:`sklearn.metrics.roc_auc_score` with `average=macro`. Refer to the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html>`__
    for more details.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the decision accuracy
    """
    return roc_auc_score(y_true, y_pred, average="macro")  # type: ignore[no-any-return]


def decision_roc_auc(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    r"""
    Calculate ROC AUC for multiclass and multilabel classification.

    The ROC AUC measures the ability of a model to distinguish between classes.
    It is calculated as the area under the curve of the true positive rate (TPR)
    against the false positive rate (FPR) at various threshold settings.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the decision ROC AUC
    """
    y_true_, y_pred_ = transform(y_true, y_pred)
    if y_pred_.ndim == y_true_.ndim == 1:
        return _decision_roc_auc_multiclass(y_true_, y_pred_)
    if y_pred_.ndim == y_true_.ndim == 2:  # noqa: PLR2004
        return _decision_roc_auc_multilabel(y_true_, y_pred_)
    msg = "Something went wrong with labels dimensions"
    logger.error(msg)
    raise ValueError(msg)


def decision_precision(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    r"""
    Calculate decision precision. Supports both multiclass and multilabel.

    This function internally uses :func:`sklearn.metrics.precision_score` with `average=macro`. Refer to the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`__
    for more details.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the decision precision
    """
    return precision_score(y_true, y_pred, average="macro")  # type: ignore[no-any-return]


def decision_recall(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    r"""
    Calculate decision recall. Supports both multiclass and multilabel.

    This function internally uses :func:`sklearn.metrics.recall_score` with `average=macro`. Refer to the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html>`__
    for more details.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the decision recall
    """
    return recall_score(y_true, y_pred, average="macro")  # type: ignore[no-any-return]


def decision_f1(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    r"""
    Calculate decision f1 score. Supports both multiclass and multilabel.

    This function internally uses :func:`sklearn.metrics.f1_score` with `average=macro`. Refer to the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`__
    for more details.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the decision accuracy
    """
    return f1_score(y_true, y_pred, average="macro")  # type: ignore[no-any-return]
