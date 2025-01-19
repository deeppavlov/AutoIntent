"""Type definitions for metrics module."""

from typing import Any

import numpy.typing as npt

from autointent.custom_types import LabelType

LABELS_VALUE_TYPE = list[LabelType] | npt.NDArray[Any]

CANDIDATE_TYPE = list[list[LabelType]] | npt.NDArray[Any]

SCORES_VALUE_TYPE = list[list[float]] | npt.NDArray[Any]
