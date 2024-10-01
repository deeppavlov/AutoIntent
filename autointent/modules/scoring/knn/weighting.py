from typing import Literal

import numpy as np

from .count_neighbors import get_counts, get_counts_multilabel


def apply_weights(
    labels: np.ndarray,
    distances: np.ndarray,
    weights: Literal["uniform", "distance", "closest"],
    n_classes: int,
    multilabel: bool,
):
    """
    Calculate probabilities

    Arguments
    ---

    `labels`:
    - multiclass case: np.ndarray of shape (n_samples, n_neighbors) with integer labels from [0,n_classes-1]
    - multilabel case: np.ndarray of shape (n_samples, n_neighbors, n_classes) with binary labels

    `distances`: np.ndarray of shape (n_samples, n_neighbors) with integer labels from 0..n_classes-1

    Return
    ---
    np.ndarray of shape (n_samples, n_classes)
    """
    n_samples, n_candidates = distances.shape

    if weights == "closest":
        return closest_weighting(labels, distances, multilabel, n_classes)

    if weights == "uniform":
        weights_ = np.ones((n_samples, n_candidates))

    elif weights == "distance":
        weights_ = 1 / (distances + 1e-5)

    if multilabel:
        counts = get_counts_multilabel(labels, weights_)
        probs = counts / weights_.sum(axis=1, keepdims=True)
    else:
        counts = get_counts(labels, n_classes, weights_)
        probs = counts / counts.sum(axis=1, keepdims=True)

    return probs


def closest_weighting(labels, distances, multilabel, n_classes):
    if not multilabel:
        labels = to_onehot(labels, n_classes)
    return _closest_weighting(labels, distances)


def _closest_weighting(labels: np.ndarray, distances: np.ndarray):
    """
    Arguments
    ---
    `labels`: array of shape (n_samples, n_candidates, n_classes) with binary labels
    `distances`: array of shape (n_samples, n_candidates) with cosine distances

    Return
    ---
    array of shape (n_samples, n_classes) with probabilities
    """
    # broadcast to (n_samples, n_candidates, n_classes)
    broadcasted_similarities = np.broadcast_to(1 - distances[..., None], shape=labels.shape)
    expanded_distances_view = np.where(labels != 0, broadcasted_similarities, -1)

    # select closest candidate for each query-class pair
    similarities = np.max(expanded_distances_view, axis=1)
    return (similarities + 1) / 2  # cosine [-1,+1] -> prob [0,1]


def to_onehot(labels: np.ndarray, n_classes):
    """convert nd array of ints to (n+1)d array of zeros and ones"""
    new_shape = (*labels.shape, n_classes)
    onehot_labels = np.zeros(shape=new_shape)
    indices = (*tuple(np.indices(labels.shape)), labels)
    onehot_labels[indices] = 1
    return onehot_labels
