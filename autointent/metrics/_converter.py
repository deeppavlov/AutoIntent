"""Converter module for metrics."""

import logging
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent.custom_types import LabelType

from .custom_types import CANDIDATE_TYPE, LABELS_VALUE_TYPE, SCORES_VALUE_TYPE

logger = logging.getLogger(__name__)


def transform(
    y_true: LABELS_VALUE_TYPE,
    y_pred: LABELS_VALUE_TYPE | CANDIDATE_TYPE | SCORES_VALUE_TYPE,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Transform y_true and y_pred to numpy arrays.

    :param y_true: Y_true values
    :param y_pred: Y_pred values
    :return:
    """
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        return y_true, y_pred
    y_pred_ = np.array(_handle_oos(y_pred))
    y_true_ = np.array(_handle_oos(y_true))
    return y_true_, y_pred_

def _handle_oos(labels: list[LabelType | None]) -> list[LabelType]:
    in_domain_labels = list(filter(lambda lab: lab is not None, labels))
    if len(in_domain_labels) == len(labels):
        return labels
    multilabel = isinstance(in_domain_labels[0], list)
    if multilabel:
        func = _add_oos_multilabel
        n_classes = len(in_domain_labels[0])
    else:
        func = _add_oos_multiclass
        n_classes = len(set(in_domain_labels))
    func = partial(func, n_classes=n_classes)
    return list(filter(func, labels))

def _add_oos_multiclass(label: int | None, n_classes: int) -> int:
    if label is None:
        label = n_classes
    return label

def _add_oos_multilabel(label: list[int] | None, n_classes: int) -> list[int]:
    if label is None:
        label = [0] * n_classes
    label += [1]
    return label
