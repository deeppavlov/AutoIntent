"""Utility functions and custom exceptions for handling multilabel predictions and errors."""

from typing import Any

import numpy as np
import numpy.typing as npt

from autointent.context.data_handler import Tag


def apply_tags(labels: npt.NDArray[Any], scores: npt.NDArray[Any], tags: list[Tag]) -> npt.NDArray[Any]:
    """
    Adjust multilabel predictions based on intent class tags.

    If some intent classes share a common tag (i.e., they are mutually exclusive) and are assigned
    to the same sample, this function retains only the class with the highest score among those
    with the shared tag.

    :param labels: Array of shape (n_samples, n_classes) with binary labels (0 or 1).
    :param scores: Array of shape (n_samples, n_classes) with float values (0 to 1).
    :param tags: List of `Tag` objects, where each tag specifies mutually exclusive intent IDs.
    :return: Adjusted array of shape (n_samples, n_classes) with binary labels.
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
                # Set all other tagged indices to 0 in the result
                for idx in tag.intent_ids:
                    if idx != max_score_index:
                        res[i, idx] = 0

    return res


class WrongClassificationError(Exception):
    """
    Exception raised when a classification module is used with incompatible data.

    This error typically occurs when a multiclass module is called on multilabel data
    or vice versa.

    :param message: Error message, defaults to a standard incompatibility message.
    """

    def __init__(self, message: str = "Multiclass module is called on multilabel data or vice-versa") -> None:
        """
        Initialize the exception.

        :param message: Error message, defaults to a standard incompatibility message.
        """
        self.message = message
        super().__init__(message)


class InvalidNumClassesError(Exception):
    """
    Exception raised when the data contains an incompatible number of classes.

    This error indicates that the number of classes in the input data does not match
    the expected number of classes for the module.

    :param message: Error message, defaults to a standard class incompatibility message.
    """

    def __init__(self, message: str = "Data with incompatible number of classes was sent to module") -> None:
        """
        Initialize the exception.

        :param message: Error message, defaults to a standard incompatibility message.
        """
        self.message = message
        super().__init__(message)
