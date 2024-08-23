import numpy as np

from ..retrieval.vectordb import retrieve_candidates_labels
from .base import DataHandler, ScoringModule


class KNNScorer(ScoringModule):
    """
    TODO:
    - add weighted knn?
    """

    def __init__(self, k, device="cuda"):
        self.k = k
        self.device = device

    def fit(self, data_handler: DataHandler):
        self._collection = data_handler.get_best_collection(self.device)
        self._n_classes = data_handler.n_classes

    def predict(self, utterances: list[str]):
        labels_pred = retrieve_candidates_labels(self._collection, self.k, utterances)
        labels_pred = np.array(labels_pred)
        if not self._collection.metadata["multilabel"]:
            counts = get_counts(labels_pred, self._n_classes)
        else:
            counts = get_counts_multilabel(labels_pred)
        denom = counts.sum(axis=1, keepdims=True)
        counts = counts.astype(float)
        np.divide(counts, denom, out=counts, where=denom!=0)    # TODO: fix this workaround because zero count can mean OOS
        return counts
    
    def clear_cache(self):
        model = self._collection._embedding_function._model
        model.to(device='cpu')
        del model
        self._collection = None


def get_counts(labels, n_classes):
    """
    Arguments
    ---
    `labels`: np.ndarray of shape (n_samples, n_candidates) with integer labels from `[0,n_classes-1]`

    Return
    ---
    np.ndarray of shape (n_samples, n_classes) with statistics of how many times each class label occured in candidates
    """
    n_queries = labels.shape[0]
    labels += n_classes * np.arange(n_queries)[:, None]
    counts = np.bincount(labels.ravel(), minlength=n_classes * n_queries).reshape(
        n_queries, n_classes
    )
    return counts


def get_counts_multilabel(labels):
    """
    Arguments
    ---
    `labels`: np.ndarray of shape (n_samples, n_candidates, n_classes) with binary labels

    Return
    ---
    np.ndarray of shape (n_samples, n_classes) with statistics of how many times each class label occured in candidates
    """
    return labels.sum(axis=1)