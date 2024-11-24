"""Regexp metrics for intent recognition."""

from typing import Protocol

import numpy as np

from ._converter import transform
from ._custom_types import LABELS_VALUE_TYPE


class RegexpMetricFn(Protocol):
    """Protocol for regexp metrics."""

    def __call__(self, y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
        """
        Calculate regexp metric.

        :param y_true: True values of labels
        :param y_pred: Predicted values of labels
        :return: Score of the regexp metric
        """
        ...


def regexp_partial_accuracy(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    """
    Calculate regexp partial accuracy.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the regexp metric
    """
    y_true_, y_pred_ = transform(y_true, y_pred)
    correct = np.mean([true in pred for true, pred in zip(y_true_, y_pred_, strict=True)])
    total = y_true_.shape[0]
    if total == 0:
        return -1  # TODO think about it
    return correct / total  # type: ignore[no-any-return]


def regexp_partial_precision(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    """
    Calculate regexp partial precision.

    :param y_true: True values of labels
    :param y_pred: Predicted values of labels
    :return: Score of the regexp metric
    """
    y_true_, y_pred_ = transform(y_true, y_pred)

    correct = np.sum([true in pred for true, pred in zip(y_true_, y_pred_, strict=True)])
    total = np.sum([pred.shape[0] > 0 for pred in y_pred_])

    if total == 0:
        return -1

    return correct / total  # type: ignore[no-any-return]
