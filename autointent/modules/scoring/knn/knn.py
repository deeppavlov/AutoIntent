import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from autointent.context import Context
from autointent.context.vector_index_client import VectorIndex, VectorIndexClient
from autointent.context.vector_index_client.cache import get_db_dir
from autointent.custom_types import LABEL_TYPE, WEIGHT_TYPES, BaseMetadataDict
from autointent.modules.scoring.base import ScoringModule

from .weighting import apply_weights


class KNNScorerDumpMetadata(BaseMetadataDict):
    model_name: str
    k: int
    weights: WEIGHT_TYPES
    n_classes: int
    multilabel: bool
    db_dir: str
    device: str


class KNNScorer(ScoringModule):
    weights: WEIGHT_TYPES
    _vector_index: VectorIndex

    def __init__(
        self,
        model_name: str,
        k: int,
        weights: WEIGHT_TYPES,
        n_classes: int = 3,
        multilabel: bool = False,
        db_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
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
        if db_dir is None:
            db_dir = str(get_db_dir())
        self.model_name = model_name
        self.k = k
        self.weights = weights
        self.n_classes = n_classes
        self.multilabel = multilabel
        self.db_dir = db_dir
        self.device = device

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: int = 3,
        weights: WEIGHT_TYPES = "distance",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> Self:
        return cls(
            model_name=model_name,
            k=k,
            weights=weights,
            n_classes=context.n_classes,
            multilabel=context.multilabel,
            db_dir=str(context.db_dir),
            device=context.device,
        )

    def fit(self, utterances: list[str], labels: list[LABEL_TYPE], **kwargs: dict[str, Any]) -> None:
        vector_index_client = VectorIndexClient(self.device, self.db_dir)
        self._vector_index = vector_index_client.get_or_create_index(self.model_name, utterances, labels)
        self._vector_index.add(utterances, labels)

        self.metadata = KNNScorerDumpMetadata(
            device=self.device,
            db_dir=self.db_dir,
            n_classes=self.n_classes,
            multilabel=self.multilabel,
            model_name=self._vector_index.model_name,
            k=self.k,
            weights=self.weights,
        )

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        labels, distances, _ = self._vector_index.query(utterances, self.k)
        return apply_weights(np.array(labels), np.array(distances), self.weights, self.n_classes, self.multilabel)

    def clear_cache(self) -> None:
        self._vector_index.delete()

    def dump(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

        self._vector_index.dump(dump_dir)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            self.metadata: KNNScorerDumpMetadata = json.load(file)

        self.n_classes = self.metadata["n_classes"]
        self.multilabel = self.metadata["multilabel"]

        vector_index_client = VectorIndexClient(device=self.metadata["device"], db_dir=self.metadata["db_dir"])
        self._vector_index = vector_index_client.get_index(self.metadata["model_name"])
