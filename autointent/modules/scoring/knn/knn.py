from typing import Literal

import numpy as np

from ..base import Context, ScoringModule
from .count_neighbors import get_counts, get_counts_multilabel
from .weighting import retrieve_candidates


class KNNScorer(ScoringModule):
    def __init__(self, k: int, weights: Literal["uniform", "distance", "closest"] | bool):
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
        if isinstance(weights, bool):
            weights = "distance" if weights else "uniform"
        self.weights = weights

    def fit(self, context: Context):
        self._collection = context.get_best_collection()
        self._n_classes = context.n_classes

    def predict(self, utterances: list[str]):
        labels_pred, weights = retrieve_candidates(self._collection, self.k, utterances, self.weights)
        if not self._collection.metadata["multilabel"]:
            counts = get_counts(labels_pred, self._n_classes, weights)
        else:
            counts = get_counts_multilabel(labels_pred, weights)
        denom = counts.sum(axis=1, keepdims=True)
        counts = counts.astype(float)
        np.divide(
            counts, denom, out=counts, where=denom != 0
        )  # TODO: fix this workaround because zero count can mean OOS
        return counts

    def clear_cache(self):
        model = self._collection._embedding_function._model
        model.to(device="cpu")
        del model
        self._collection = None
