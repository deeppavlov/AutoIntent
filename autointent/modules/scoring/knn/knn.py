from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.custom_types import WEIGHT_TYPES
from autointent.modules.scoring.base import ScoringModule

from .weighting import apply_weights


class KNNScorer(ScoringModule):
    def __init__(self, k: int, weights: WEIGHT_TYPES | bool) -> None:
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

    def fit(self, context: Context) -> None:
        self._multilabel = context.multilabel
        self._collection = context.get_best_index()
        self._n_classes = context.n_classes

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        labels, distances, _ = self._collection.query(utterances, self.k)
        return apply_weights(np.array(labels), np.array(distances), self.weights, self._n_classes, self._multilabel)

    def clear_cache(self) -> None:
        pass
