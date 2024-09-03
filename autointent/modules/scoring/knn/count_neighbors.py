import numpy as np


def get_counts(labels, n_classes, weights):
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
    counts = np.bincount(labels.ravel(), minlength=n_classes * n_queries, weights=weights.ravel()).reshape(
        n_queries, n_classes
    )
    return counts


def get_counts_multilabel(labels, weights):
    """
    Arguments
    ---
    `labels`: np.ndarray of shape (n_samples, n_candidates, n_classes) with binary labels
    `weights`: np.ndarray of shape (n_samples, n_candidates) with float weights

    Return
    ---
    np.ndarray of shape (n_samples, n_classes) with statistics of how many times each class label occured in candidates
    """
    return (labels * weights).sum(axis=1)
