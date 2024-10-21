import numpy as np
import numpy.typing as npt

from autointent.custom_types import LABEL_TYPE

LABELS_VALUE_TYPE = list[LABEL_TYPE] | npt.NDArray[np.int64]

CANDIDATE_TYPE = list[list[LABEL_TYPE]] | npt.NDArray[np.int64]

SCORES_VALUE_TYPE = list[list[float]] | npt.NDArray[np.float64]
