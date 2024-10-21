from typing import Protocol

import numpy as np

from autointent.metrics.converter import transform

from .custom_types import LABELS_VALUE_TYPE


class RegexpMetricFn(Protocol):
    def __call__(self, y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float: ...


def regexp_partial_accuracy(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    y_true_, y_pred_ = transform(y_true, y_pred)
    correct = np.mean([true in pred for true, pred in zip(y_true_, y_pred_, strict=True)])
    total = y_true_.shape[0]
    if total == 0:
        return -1  # TODO think about it
    return correct / total  # type: ignore[no-any-return]


def regexp_partial_precision(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    y_true_, y_pred_ = transform(y_true, y_pred)

    correct = np.sum([true in pred for true, pred in zip(y_true_, y_pred_, strict=True)])
    total = np.sum([pred.shape[0] > 0 for pred in y_pred_])

    if total == 0:
        return -1

    return correct / total  # type: ignore[no-any-return]
