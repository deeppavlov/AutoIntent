import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

from autointent.context import Context
from autointent.context.vector_index_client import VectorIndex, VectorIndexClient
from autointent.custom_types import WEIGHT_TYPES
from autointent.modules.scoring.base import ScoringModule

from .weighting import apply_weights


class KNNScorerDumpMetadata(TypedDict):
    device: str
    db_dir: str
    n_classes: int
    multilabel: bool
    model_name: str


class KNNScorer(ScoringModule):
    weights: WEIGHT_TYPES
    metadata_dict_name: str = "metadata.json"
    _vector_index: VectorIndex

    def __init__(self, k: int, weights: WEIGHT_TYPES) -> None:
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
        self.weights = weights

    def fit(self, context: Context) -> None:
        self._multilabel = context.multilabel
        self._vector_index = context.get_best_index()
        self._n_classes = context.n_classes

        self.metadata = KNNScorerDumpMetadata(
            device=context.device,
            db_dir=context.db_dir,
            n_classes=self._n_classes,
            multilabel=self._multilabel,
            model_name=self._vector_index.model_name,
        )

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        labels, distances, _ = self._vector_index.query(utterances, self.k)
        return apply_weights(np.array(labels), np.array(distances), self.weights, self._n_classes, self._multilabel)

    def clear_cache(self) -> None:
        self._vector_index.delete()

    def dump(self, path: str) -> None:
        dump_dir = Path(path)
        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            self.metadata = json.load(file)

        self._n_classes = self.metadata["n_classes"]
        self._multilabel = self.metadata["multilabel"]

        vector_index_client = VectorIndexClient(device=self.metadata["device"], db_dir=self.metadata["db_dir"])
        self._vector_index = vector_index_client.get_index(self.metadata["model_name"])
