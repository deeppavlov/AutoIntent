"""Utility functions for calculating probabilities and weighting nearest neighbors."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from autointent.custom_types import WEIGHT_TYPES

from .count_neighbors import get_counts, get_counts_multilabel


def apply_weights(
    labels: NDArray[Any],
    distances: NDArray[Any],
    weights: WEIGHT_TYPES,
    n_classes: int,
    multilabel: bool,
) -> NDArray[Any]:
    """
    Calculate probabilities based on labels, distances, and weighting strategy.

    :param labels:
        - For multiclass: Array of shape (n_samples, n_neighbors) with integer labels in [0, n_classes - 1].
        - For multilabel: Array of shape (n_samples, n_neighbors, n_classes) with binary labels.
    :param distances: Array of shape (n_samples, n_neighbors) with float distances.
    :param weights: Weighting strategy to apply. Options are "closest", "uniform", or "distance".
    :param n_classes: Number of classes in the dataset.
    :param multilabel: Whether the task is multilabel classification.
    :return: Array of shape (n_samples, n_classes) with calculated probabilities.
    """
    n_samples, n_candidates = distances.shape

    if weights == "closest":
        return closest_weighting(labels, distances, multilabel, n_classes)

    if weights == "uniform":
        weights_: NDArray[Any] = np.ones((n_samples, n_candidates))

    elif weights == "distance":
        weights_ = 1 / (distances + 1e-5)

    if multilabel:
        counts = get_counts_multilabel(labels, weights_)
        probs = counts / weights_.sum(axis=1, keepdims=True)
    else:
        counts = get_counts(labels, n_classes, weights_)  # type: ignore[assignment]
        probs = counts / counts.sum(axis=1, keepdims=True)

    return probs  # type: ignore[no-any-return]


def closest_weighting(labels: NDArray[Any], distances: NDArray[Any], multilabel: bool, n_classes: int) -> NDArray[Any]:
    """
    Apply closest weighting strategy.

    :param labels:
        - For multiclass: Array of shape (n_samples, n_neighbors) with integer labels in [0, n_classes - 1].
        - For multilabel: Array of shape (n_samples, n_neighbors, n_classes) with binary labels.
    :param distances: Array of shape (n_samples, n_neighbors) with cosine distances.
    :param multilabel: Whether the task is multilabel classification.
    :param n_classes: Number of classes in the dataset.
    :return: Array of shape (n_samples, n_classes) with calculated probabilities.
    """
    if not multilabel:
        labels = to_onehot(labels, n_classes)
    return _closest_weighting(labels, distances)


def _closest_weighting(labels: NDArray[Any], distances: NDArray[Any]) -> NDArray[Any]:
    """
    Apply closest weighting strategy for multilabel classification.

    :param labels: Array of shape (n_samples, n_candidates, n_classes) with binary labels.
    :param distances: Array of shape (n_samples, n_candidates) with cosine distances.
    :return: Array of shape (n_samples, n_classes) with calculated probabilities.
    """
    # Broadcast to (n_samples, n_candidates, n_classes)
    broadcasted_similarities = np.broadcast_to(1 - distances[..., None], shape=labels.shape)
    expanded_distances_view = np.where(labels != 0, broadcasted_similarities, -1)

    # Select closest candidate for each query-class pair
    similarities = np.max(expanded_distances_view, axis=1)
    return (similarities + 1) / 2  # type: ignore[no-any-return]  # cosine [-1,+1] -> prob [0,1]


def to_onehot(labels: NDArray[Any], n_classes: int) -> NDArray[Any]:
    """
    Convert an array of integer labels to a one-hot encoded array.

    :param labels: Array of shape (n_samples, n_neighbors) with integer labels.
    :param n_classes: Number of classes in the dataset.
    :return: One-hot encoded array of shape (n_samples, n_neighbors, n_classes).
    """
    new_shape = (*labels.shape, n_classes)
    onehot_labels = np.zeros(shape=new_shape)
    indices = (*tuple(np.indices(labels.shape)), labels)
    onehot_labels[indices] = 1
    return onehot_labels
