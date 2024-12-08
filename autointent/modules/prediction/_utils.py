"""Utility functions and custom exceptions for handling multilabel predictions and errors."""

from typing import Any

import numpy as np
import numpy.typing as npt

from autointent.schemas import Tag


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
    labels = labels.copy()

    for tag in tags:
        intent_ids = tag.intent_ids

        labels_sub = labels[:, intent_ids]
        scores_sub = scores[:, intent_ids]

        assigned = labels_sub == 1
        num_assigned = assigned.sum(axis=1)

        assigned_scores = np.where(assigned, scores_sub, -np.inf)

        samples_to_adjust = np.where(num_assigned > 1)[0]

        if samples_to_adjust.size > 0:
            assigned_scores_adjust = assigned_scores[samples_to_adjust, :]
            idx_max_adjust = assigned_scores_adjust.argmax(axis=1)

            labels_sub[samples_to_adjust, :] = 0
            labels_sub[samples_to_adjust, idx_max_adjust] = 1

        labels[:, intent_ids] = labels_sub

    return labels


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
