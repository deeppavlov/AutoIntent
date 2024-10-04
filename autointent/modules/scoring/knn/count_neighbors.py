import numpy as np
from numpy.typing import NDArray


def get_counts(
    labels: NDArray[np.int_], n_classes: int, weights: NDArray[np.float_]
) -> NDArray[np.float_]:
    """
    Arguments
    ---
    `labels`: np.ndarray of shape (n_samples, n_candidates) with integer labels from `[0,n_classes-1]`
    `weights`: np.ndarray of shape (n_samples, n_candidates) with float weights

    Return
    ---
    np.ndarray of shape (n_samples, n_classes) with statistics of how many times each class label occured in candidates
    """
    n_queries = labels.shape[0]
    labels += n_classes * np.arange(n_queries)[:, None]
    return np.bincount(labels.ravel(), minlength=n_classes * n_queries, weights=weights.ravel()).reshape(
        n_queries, n_classes
    )


def get_counts_multilabel(
    labels: NDArray[np.int_], weights: NDArray[np.float_]
) -> NDArray[np.float_]:
    """
    Arguments
    ---
    `labels`: np.ndarray of shape (n_samples, n_candidates, n_classes) with binary labels
    `weights`: np.ndarray of shape (n_samples, n_candidates) with float weights

    Return
    ---
    np.ndarray of shape (n_samples, n_classes) with statistics of how many times each class label occured in candidates
    """
    return (labels * weights[..., None]).sum(axis=1)
