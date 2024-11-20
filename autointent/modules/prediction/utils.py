from typing import Any

import numpy as np
import numpy.typing as npt

from autointent.context.data_handler import Tag


def apply_tags(labels: npt.NDArray[Any], scores: npt.NDArray[Any], tags: list[Tag]) -> npt.NDArray[Any]:
    """
    this function is intended to be used with multilabel predictor

    If some intent classes have common tag (i.e. they are mutually exclusive) \
    and were assigned to one sample, leave only that class that has the highest score.

    Arguments
    ---
    - `labels`: np.ndarray of shape (n_samples, n_classes) with binary labels
    - `scores`: np.ndarray of shape (n_samples, n_classes) with float values from 0..1
    - `tags`: list of Tags

    Return
    ---
    np.ndarray of shape (n_samples, n_classes) with binary labels
    """

    n_samples, _ = labels.shape
    res = np.copy(labels)

    for i in range(n_samples):
        sample_labels = labels[i].astype(bool)
        sample_scores = scores[i]

        for tag in tags:
            if any(sample_labels[idx] for idx in tag.intent_ids):
                # Find the index of the class with the highest score among the tagged indices
                max_score_index = max(tag.intent_ids, key=lambda idx: sample_scores[idx])
                # Set all other tagged indices to 0 in the res
                for idx in tag.intent_ids:
                    if idx != max_score_index:
                        res[i, idx] = 0

    return res


class WrongClassificationError(Exception):
    def __init__(self, message: str = "multiclass module is called on multilabel data or vice-versa") -> None:
        self.message = message
        super().__init__(message)


class InvalidNumClassesError(Exception):
    def __init__(self, message: str = "data with incompatible number of classes was sent to module") -> None:
        self.message = message
        super().__init__(message)
