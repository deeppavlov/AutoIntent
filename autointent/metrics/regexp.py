from typing import Protocol

import numpy as np
import numpy.typing as npt

from autointent.metrics.converter import transform

TRUE_VALUE_TYPE = list[int] | npt.NDArray[np.int64]
PREDICTED_VALUE_TYPE = list[list[int]] | npt.NDArray[np.int64]


class RegexpMetricFn(Protocol):
    def __call__(self, y_true: TRUE_VALUE_TYPE, y_pred: PREDICTED_VALUE_TYPE) -> float: ...


def regexp_partial_accuracy(y_true: TRUE_VALUE_TYPE, y_pred: PREDICTED_VALUE_TYPE) -> float:
    y_true_, y_pred_ = transform(y_true, y_pred)
    correct = np.mean([true in pred for true, pred in zip(y_true_, y_pred_, strict=True)])
    total = y_true_.shape[0]
    if total == 0:
        return -1  # TODO think about it
    return correct / total  # type: ignore[no-any-return]


def regexp_partial_precision(y_true: TRUE_VALUE_TYPE, y_pred: PREDICTED_VALUE_TYPE) -> float:
    y_true_, y_pred_ = transform(y_true, y_pred)

    correct = np.sum([true in pred for true, pred in zip(y_true_, y_pred_, strict=True)])
    total = np.sum([pred.shape[0] > 0 for pred in y_pred_])

    if total == 0:
        return -1

    return correct / total  # type: ignore[no-any-return]
