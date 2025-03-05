"""Type definitions for metrics module."""

from typing import Any

import numpy.typing as npt

from autointent.custom_types import ListOfLabels

LABELS_VALUE_TYPE = ListOfLabels

CANDIDATE_TYPE = list[ListOfLabels] | npt.NDArray[Any]

SCORES_VALUE_TYPE = list[list[float]] | npt.NDArray[Any]
