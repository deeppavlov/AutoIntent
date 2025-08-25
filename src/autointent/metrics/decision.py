"""Prediction metrics for multiclass and multilabel classification tasks."""

import logging
from functools import partial
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from autointent.custom_types import ListOfGenericLabels, ListOfLabels

from ._converter import transform

logger = logging.getLogger(__name__)


class DecisionMetricFn(Protocol):
    """Protocol for decision metrics."""

    def __call__(self, y_true: ListOfGenericLabels, y_pred: ListOfGenericLabels) -> float:
        """Calculate decision metric.

        Args:
            y_true: True values of labels
                - multiclass case: list representing an array shape `(n_samples,)` of integer class labels
                - multilabel case: list representing a matrix of shape `(n_samples, n_classes)` with binary values
            y_pred: Predicted values of labels. Same shape as `y_true`
        Returns:
            Score of the decision metric
        """
        ...


def handle_oos(y_true: ListOfGenericLabels, y_pred: ListOfGenericLabels) -> tuple[ListOfLabels, ListOfLabels]:
    """Convert labels of OOS samples to make them usable in decision metrics.

    Args:
        y_true: True values of labels
        y_pred: Predicted values of labels

    Returns:
        Tuple of transformed true and predicted labels
    """
    in_domain_labels = list(filter(lambda lab: lab is not None, y_true))
    if isinstance(in_domain_labels[0], list):
        func = _add_oos_multilabel
        n_classes = len(in_domain_labels[0])
    else:
        func = _add_oos_multiclass  # type: ignore[assignment]
        n_classes = len(set(in_domain_labels))
    func = partial(func, n_classes=n_classes)
    return list(map(func, y_true)), list(map(func, y_pred))


def _add_oos_multiclass(label: int | None, n_classes: int) -> int:
    """Add OOS label for multiclass classification.

    Args:
        label: Original label
        n_classes: Number of classes

    Returns:
        Transformed label
    """
    if label is None:
        return n_classes
    return label


def _add_oos_multilabel(label: list[int] | None, n_classes: int) -> list[int]:
    """Add OOS label for multilabel classification.

    Args:
        label: Original label
        n_classes: Number of classes

    Returns:
        Transformed label
    """
    if label is None:
        return [0] * n_classes + [1]
    return [*label, 1]


def decision_accuracy(y_true: ListOfGenericLabels, y_pred: ListOfGenericLabels) -> float:
    r"""Calculate decision accuracy. Supports both multiclass and multilabel.

    The decision accuracy is calculated as:

    .. math::

        \text{Accuracy} = \frac{\sum_{i=1}^N \mathbb{1}(y_{\text{true},i} = y_{\text{pred},i})}{N}

    where:

    - :math:`N` is the total number of samples,
    - :math:`y_{\text{true},i}` is the true label for the :math:`i`-th sample,
    - :math:`y_{\text{pred},i}` is the predicted label for the :math:`i`-th sample,
    - :math:`\mathbb{1}(\text{condition})` is the indicator function that equals 1 if the condition
    is true and 0 otherwise.

    Args:
        y_true: True values of labels
        y_pred: Predicted values of labels

    Returns:
        Score of the decision accuracy
    """
    y_true_, y_pred_ = transform(*handle_oos(y_true, y_pred))
    return float(np.mean(y_true_ == y_pred_))


def _decision_roc_auc_multiclass(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
    r"""Calculate ROC AUC for multiclass.

    The ROC AUC score for multiclass is calculated as the mean ROC AUC score
    across all classes, where each class is treated as a binary classification task
    (one-vs-rest).

    .. math::

        \text{ROC AUC}_{\text{multiclass}} = \frac{1}{K} \sum_{k=1}^K \text{ROC AUC}_k

    where:

    - :math:`K` is the number of classes,
    - :math:`\text{ROC AUC}_k` is the ROC AUC score for the :math:`k`-th class,
    calculated by treating it as a binary classification problem (class :math:`k` vs rest).

    Args:
        y_true: True values of labels
        y_pred: Predicted values of labels

    Returns:
        Score of the decision ROC AUC
    """
    n_classes = len(np.unique(y_true))
    roc_auc_scores: list[float] = []
    for k in range(n_classes):
        binarized_true = (y_true == k).astype(int)
        binarized_pred = (y_pred == k).astype(int)
        roc_auc_scores.append(roc_auc_score(binarized_true, binarized_pred))

    return float(np.mean(roc_auc_scores))


def _decision_roc_auc_multilabel(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
    r"""Calculate ROC AUC for multilabel.

    This function internally uses :func:`sklearn.metrics.roc_auc_score` with `average=macro`. Refer to the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html>`__
    for more details.

    Args:
        y_true: True values of labels
        y_pred: Predicted values of labels

    Returns:
        Score of the decision ROC AUC
    """
    return float(roc_auc_score(y_true, y_pred, average="macro"))


def decision_roc_auc(y_true: ListOfGenericLabels, y_pred: ListOfGenericLabels) -> float:
    r"""Calculate ROC AUC for multiclass and multilabel classification.

    The ROC AUC measures the ability of a model to distinguish between classes.
    It is calculated as the area under the curve of the true positive rate (TPR)
    against the false positive rate (FPR) at various threshold settings.

    Args:
        y_true: True values of labels
        y_pred: Predicted values of labels

    Returns:
        Score of the decision ROC AUC
    """
    y_true_, y_pred_ = transform(*handle_oos(y_true, y_pred))
    if y_pred_.ndim == y_true_.ndim == 1:
        return _decision_roc_auc_multiclass(y_true_, y_pred_)
    if y_pred_.ndim == y_true_.ndim == 2:  # noqa: PLR2004
        # not working with 1 class in y_true
        return _decision_roc_auc_multilabel(y_true_, y_pred_)
    msg = "Something went wrong with labels dimensions"
    logger.error(msg)
    raise ValueError(msg)


def decision_precision(y_true: ListOfGenericLabels, y_pred: ListOfGenericLabels) -> float:
    r"""Calculate decision precision. Supports both multiclass and multilabel.

    This function internally uses :func:`sklearn.metrics.precision_score` with `average=macro`. Refer to the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`__
    for more details.

    Args:
        y_true: True values of labels
        y_pred: Predicted values of labels

    Returns:
        Score of the decision precision
    """
    return float(precision_score(*handle_oos(y_true, y_pred), average="macro"))


def decision_recall(y_true: ListOfGenericLabels, y_pred: ListOfGenericLabels) -> float:
    r"""Calculate decision recall. Supports both multiclass and multilabel.

    This function internally uses :func:`sklearn.metrics.recall_score` with `average=macro`. Refer to the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html>`__
    for more details.

    Args:
        y_true: True values of labels
        y_pred: Predicted values of labels

    Returns:
        Score of the decision recall
    """
    return float(recall_score(*handle_oos(y_true, y_pred), average="macro"))


def decision_f1(y_true: ListOfGenericLabels, y_pred: ListOfGenericLabels) -> float:
    r"""Calculate decision F1 score. Supports both multiclass and multilabel.

    This function internally uses :func:`sklearn.metrics.f1_score` with `average=macro`. Refer to the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`__
    for more details.

    Args:
        y_true: True values of labels
        y_pred: Predicted values of labels

    Returns:
        Score of the decision F1 score
    """
    return float(f1_score(*handle_oos(y_true, y_pred), average="macro"))
