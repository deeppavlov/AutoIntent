"""Utility functions for handling multilabel predictions."""

from typing import Any

import numpy as np
import numpy.typing as npt

from autointent.schemas import Tag


def apply_tags(labels: npt.NDArray[Any], scores: npt.NDArray[Any], tags: list[Tag]) -> npt.NDArray[Any]:
    """Adjust multilabel predictions based on intent class tags.

    If some intent classes share a common tag (i.e., they are mutually exclusive) and are assigned
    to the same sample, this function retains only the class with the highest score among those
    with the shared tag.

    Args:
        labels: Array of shape (n_samples, n_classes) with binary labels (0 or 1)
        scores: Array of shape (n_samples, n_classes) with float values (0 to 1)
        tags: List of Tag objects, where each tag specifies mutually exclusive intent IDs

    Returns:
        Array of shape (n_samples, n_classes) with adjusted binary labels

    Examples:
        >>> import numpy as np
        >>> from autointent.schemas import Tag
        >>> labels = np.array([[1, 1, 0], [1, 1, 1]])
        >>> scores = np.array([[0.8, 0.6, 0.3], [0.7, 0.9, 0.5]])
        >>> tags = [Tag(name="group1", intent_ids=[0, 1])]
        >>> adjusted = apply_tags(labels, scores, tags)
        >>> print(adjusted)
        [[1 0 0]
         [0 1 1]]
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
