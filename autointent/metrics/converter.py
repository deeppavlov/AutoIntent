import logging
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def transform(
    y_true: list[int] | list[list[int]] | npt.NDArray[np.int64],
    y_pred: list[int] | list[list[int]] | list[list[list[int]]] | list[list[float]] | npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        return y_true, y_pred
    y_pred_ = np.array(y_pred)
    y_true_ = np.array(y_true)
    return y_true_, y_pred_
