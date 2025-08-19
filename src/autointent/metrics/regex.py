"""Metrics for regex modules."""

from typing import Protocol

import numpy as np

from ._converter import transform
from .custom_types import LABELS_VALUE_TYPE


class RegexMetricFn(Protocol):
    """Protocol for regex metrics."""

    def __call__(self, y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
        """Calculate regex metric.

        Args:
            y_true: True values of labels.
            y_pred: Predicted values of labels.

        Returns:
            Score of the regex metric.
        """
        ...


def regex_partial_accuracy(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    r"""Calculate regex partial accuracy.

    The regex partial accuracy is calculated as:

    .. math::

        \text{Partial Accuracy} = \frac{\sum_{i=1}^N \mathbb{1}(y_{\text{true},i} \in y_{\text{pred},i})}{N}

    where:

    - :math:`N` is the total number of samples,
    - :math:`y_{\text{true},i}` is the true label for the :math:`i`-th sample,
    - :math:`y_{\text{pred},i}` is the predicted label for the :math:`i`-th sample,
    - :math:`\mathbb{1}(\text{condition})` is the indicator function that equals 1 if the condition
      is true and 0 otherwise.

    Args:
        y_true: True values of labels.
        y_pred: Predicted values of labels.

    Returns:
        Score of the regex partial accuracy.
    """
    y_true_, y_pred_ = transform(y_true, y_pred)
    correct = np.mean([true in pred for true, pred in zip(y_true_, y_pred_, strict=True)])
    total = y_true_.shape[0]
    if total == 0:
        return -1  # TODO: think about it
    return float(correct / total)


def regex_partial_precision(y_true: LABELS_VALUE_TYPE, y_pred: LABELS_VALUE_TYPE) -> float:
    r"""Calculate regex partial precision.

    The regex partial precision is calculated as:

    .. math::

        \text{Partial Precision} = \frac{\sum_{i=1}^N \mathbb{1}(y_{\text{true},i} \in y_{\text{pred},i})}{\sum_{i=1}^N
        \mathbb{1}(|y_{\text{pred},i}| > 0)}

    where:

    - :math:`N` is the total number of samples,
    - :math:`y_{\text{true},i}` is the true label for the :math:`i`-th sample,
    - :math:`y_{\text{pred},i}` is the predicted label for the :math:`i`-th sample,
    - :math:`|y_{\text{pred},i}|` is the number of predicted labels for the :math:`i`-th sample,
    - :math:`\mathbb{1}(\text{condition})` is the indicator function that equals 1 if the condition
      is true and 0 otherwise.

    Args:
        y_true: True values of labels.
        y_pred: Predicted values of labels.

    Returns:
        Score of the regex partial precision.
    """
    y_true_, y_pred_ = transform(y_true, y_pred)

    correct = np.sum([true in pred for true, pred in zip(y_true_, y_pred_, strict=True)])
    total = np.sum([pred.shape[0] > 0 for pred in y_pred_])

    if total == 0:
        return -1

    return correct / total  # type: ignore[no-any-return]
