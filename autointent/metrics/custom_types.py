"""Type definitions for metrics module."""
import numpy as np
import numpy.typing as npt

from autointent.custom_types import LabelType

LABELS_VALUE_TYPE = list[LabelType] | npt.NDArray[np.int64]

CANDIDATE_TYPE = list[list[LabelType]] | npt.NDArray[np.int64]

SCORES_VALUE_TYPE = list[list[float]] | npt.NDArray[np.float64]
