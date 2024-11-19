"""Converter module for metrics."""
import logging
from typing import Any

import numpy as np
import numpy.typing as npt

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
    y_pred_ = np.array(y_pred)
    y_true_ = np.array(y_true)
    return y_true_, y_pred_
