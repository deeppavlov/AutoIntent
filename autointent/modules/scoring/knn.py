from typing import Literal
import numpy as np

from ..retrieval.vectordb import retrieve_candidates
from .base import DataHandler, ScoringModule


class KNNScorer(ScoringModule):
    def __init__(self, k: int, weights: Literal["uniform", "distance", "closest"] | bool, device: str = "cuda:0"):
        """
        Arguments
        ---
        - `k`: int, number of closest neighbors to consider during inference;
        - `weights`: bool or str from "uniform", "distance", "closest"
            - uniform (equivalent to False): a unit weight for each sample
            - distance (equivalent to True): weight is calculated as 1 / (distance_to_neighbor + 1e-5),
            - closest: each sample has a non zero weight iff is the closest sample of some class
        - `device`: str, something like "cuda:0" or "cuda:0,1,2", a device to store embedding function
        """
        self.k = k
        self.device = device
        if isinstance(weights, bool):
            weights = "distance" if weights else "uniform"
        self.weights = weights

    def fit(self, data_handler: DataHandler):
        self._collection = data_handler.get_best_collection(self.device)
        self._n_classes = data_handler.n_classes

    def predict(self, utterances: list[str]):
        labels_pred, weights = retrieve_candidates(self._collection, self.k, utterances, self.weights)
        if not self._collection.metadata["multilabel"]:
            counts = get_counts(labels_pred, self._n_classes, weights)
        else:
            counts = get_counts_multilabel(labels_pred, weights)
        denom = counts.sum(axis=1, keepdims=True)
        counts = counts.astype(float)
        np.divide(counts, denom, out=counts, where=denom!=0)    # TODO: fix this workaround because zero count can mean OOS
        return counts
    
    def clear_cache(self):
        model = self._collection._embedding_function._model
        model.to(device='cpu')
        del model
        self._collection = None


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