"""Util functions for counting neighbors in kNN-based models."""
import numpy as np
from numpy.typing import NDArray


def get_counts(labels: NDArray[np.int_], n_classes: int, weights: NDArray[np.float64]) -> NDArray[np.int64]:
    """
    Get counts of labels in candidates for multiclass classification.

    :param labels: np.ndarray of shape (n_samples, n_candidates) with integer labels from `[0,n_classes-1]`
    :param n_classes: number of classes
    :param weights: np.ndarray of shape (n_samples, n_candidates) with integer labels from `[0,n_classes-1]`
    :return: np.ndarray of shape (n_samples, n_classes) with statistics of how many times each class
        label occured in candidates
    """
    n_queries = labels.shape[0]
    labels += n_classes * np.arange(n_queries)[:, None]
    return np.bincount(labels.ravel(), minlength=n_classes * n_queries, weights=weights.ravel()).reshape(
        n_queries, n_classes
    )


def get_counts_multilabel(labels: NDArray[np.int_], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Get counts of labels in candidates for multilabel classification.

    :param labels: np.ndarray of shape (n_samples, n_candidates, n_classes) with binary labels
    :param weights: np.ndarray of shape (n_samples, n_candidates) with float weights
    :return: np.ndarray of shape (n_samples, n_classes) with statistics of how many times
            each class label occured in candidates
    """
    return (labels * weights[..., None]).sum(axis=1)  # type: ignore[no-any-return]
