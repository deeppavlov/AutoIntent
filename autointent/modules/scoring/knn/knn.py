import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from autointent.context import Context
from autointent.context.vector_index_client import VectorIndex, VectorIndexClient
from autointent.context.vector_index_client.cache import get_db_dir
from autointent.custom_types import WEIGHT_TYPES, BaseMetadataDict, LabelType
from autointent.modules.scoring.base import ScoringModule

from .weighting import apply_weights


class KNNScorerDumpMetadata(BaseMetadataDict):
    n_classes: int
    multilabel: bool
    db_dir: str
    batch_size: int
    max_length: int | None


class KNNScorer(ScoringModule):
    weights: WEIGHT_TYPES
    _vector_index: VectorIndex
    prebuilt_index: bool = False

    def __init__(
        self,
        model_name: str,
        k: int,
        weights: WEIGHT_TYPES,
        db_dir: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int | None = None,
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
        self.db_dir = db_dir
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: int,
        weights: WEIGHT_TYPES,
        model_name: str | None = None,
    ) -> Self:
        if model_name is None:
            model_name = context.optimization_info.get_best_embedder()
            prebuilt_index = True
        else:
            prebuilt_index = context.vector_index_client.exists(model_name)

        instance = cls(
            model_name=model_name,
            k=k,
            weights=weights,
            db_dir=str(context.db_dir),
            device=context.device,
            batch_size=context.embedder_batch_size,
            max_length=context.embedder_max_length,
        )
        instance.prebuilt_index = prebuilt_index
        return instance

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        if isinstance(labels[0], list):
            self.n_classes = len(labels[0])
            self.multilabel = True
        else:
            self.n_classes = len(set(labels))
            self.multilabel = False
        vector_index_client = VectorIndexClient(self.device, self.db_dir)

        if self.prebuilt_index:
            # this happens only after RetrievalNode optimization
            self._vector_index = vector_index_client.get_index(self.model_name)
            if len(utterances) != len(self._vector_index.texts):
                msg = "Vector index mismatches provided utterances"
                raise ValueError(msg)
        else:
            self._vector_index = vector_index_client.create_index(self.model_name, utterances, labels)

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        labels, distances, _ = self._vector_index.query(utterances, self.k)
        return apply_weights(np.array(labels), np.array(distances), self.weights, self.n_classes, self.multilabel)

    def clear_cache(self) -> None:
        self._vector_index.delete()

    def dump(self, path: str) -> None:
        self.metadata = KNNScorerDumpMetadata(
            db_dir=self.db_dir,
            n_classes=self.n_classes,
            multilabel=self.multilabel,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

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

        vector_index_client = VectorIndexClient(
            device=self.device,
            db_dir=self.metadata["db_dir"],
            embedder_batch_size=self.metadata["batch_size"],
            embedder_max_length=self.metadata["max_length"],
        )
        self._vector_index = vector_index_client.get_index(self.model_name)
