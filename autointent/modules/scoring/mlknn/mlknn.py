import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from autointent import Context
from autointent.context.vector_index_client import VectorIndexClient
from autointent.context.vector_index_client.cache import get_db_dir
from autointent.custom_types import BaseMetadataDict, LabelType
from autointent.modules.scoring.base import ScoringModule


class MLKnnScorerDumpMetadata(BaseMetadataDict):
    db_dir: str
    n_classes: int
    batch_size: int
    max_length: int | None


class ArrayToSave(TypedDict):
    prior_prob_true: NDArray[np.float64]
    prior_prob_false: NDArray[np.float64]
    cond_prob_true: NDArray[np.float64]
    cond_prob_false: NDArray[np.float64]


class MLKnnScorer(ScoringModule):
    _multilabel: bool
    n_classes: int
    _prior_prob_true: NDArray[np.float64]
    _prior_prob_false: NDArray[np.float64]
    _cond_prob_true: NDArray[np.float64]
    _cond_prob_false: NDArray[np.float64]
    arrays_filename: str = "probs.npz"
    metadata: MLKnnScorerDumpMetadata
    prebuilt_index: bool = False
    name = "mlknn"

    def __init__(
        self,
        k: int,
        model_name: str,
        db_dir: str | None = None,
        s: float = 1.0,
        ignore_first_neighbours: int = 0,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int | None = None,
    ) -> None:
        if db_dir is None:
            db_dir = str(get_db_dir())
        self.k = k
        self.model_name = model_name
        self.s = s
        self.ignore_first_neighbours = ignore_first_neighbours
        self.db_dir = db_dir
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: int,
        s: float = 1.0,
        ignore_first_neighbours: int = 0,
        model_name: str | None = None,
    ) -> Self:
        if model_name is None:
            model_name = context.optimization_info.get_best_embedder()
            prebuilt_index = True
        else:
            prebuilt_index = context.vector_index_client.exists(model_name)

        instance = cls(
            k=k,
            model_name=model_name,
            s=s,
            ignore_first_neighbours=ignore_first_neighbours,
            db_dir=str(context.get_db_dir()),
            device=context.get_device(),
            batch_size=context.get_batch_size(),
            max_length=context.get_max_length(),
        )
        instance.prebuilt_index = prebuilt_index
        return instance

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        if not isinstance(labels[0], list):
            msg = "mlknn scorer support only multilabel input"
            raise TypeError(msg)

        self.n_classes = len(labels[0])

        vector_index_client = VectorIndexClient(self.device, self.db_dir)

        if self.prebuilt_index:
            # this happens only when LinearScorer is within Pipeline opimization after RetrievalNode optimization
            self.vector_index = vector_index_client.get_index(self.model_name)
            if len(utterances) != len(self.vector_index.texts):
                msg = "Vector index mismatches provided utterances"
                raise ValueError(msg)
        else:
            self.vector_index = vector_index_client.create_index(self.model_name, utterances, labels)

        self.features = (
            self.vector_index.embedder.embed(utterances)
            if self.vector_index.is_empty()
            else self.vector_index.get_all_embeddings()
        )
        self.labels = np.array(labels)
        self._prior_prob_true, self._prior_prob_false = self._compute_prior(self.labels)
        self._cond_prob_true, self._cond_prob_false = self._compute_cond()

    def _compute_prior(self, y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        prior_prob_true = (self.s + y.sum(axis=0)) / (self.s * 2 + y.shape[0])
        prior_prob_false = 1 - prior_prob_true
        return prior_prob_true, prior_prob_false

    def _compute_cond(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        c = np.zeros((self.n_classes, self.k + 1), dtype=int)
        cn = np.zeros((self.n_classes, self.k + 1), dtype=int)

        neighbors_labels = self._get_neighbors(self.features)

        for i in range(self.labels.shape[0]):
            deltas = np.sum(neighbors_labels[i], axis=0).astype(int)
            idx_helper = np.arange(self.n_classes)
            deltas_idx = deltas[idx_helper]
            c[idx_helper, deltas_idx] += self.labels[i]
            cn[idx_helper, deltas_idx] += 1 - self.labels[i]

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_prob_true = (self.s + c) / (self.s * (self.k + 1) + c_sum[:, None])
        cond_prob_false = (self.s + cn) / (self.s * (self.k + 1) + cn_sum[:, None])

        return cond_prob_true, cond_prob_false

    def _get_neighbors(self, queries: list[str] | NDArray[Any]) -> NDArray[np.int64]:
        """
        retrieve nearest neighbors and return their labels in binary format

        Return
        ---
        array of shape (n_queries, n_candidates, n_classes)
        """
        labels, _, _ = self.vector_index.query(
            queries,
            self.k + self.ignore_first_neighbours,
        )
        return np.array([candidates[self.ignore_first_neighbours :] for candidates in labels])

    def predict_labels(self, utterances: list[str], thresh: float = 0.5) -> NDArray[np.int64]:
        probas = self.predict(utterances)
        return (probas > thresh).astype(int)

    def predict(self, utterances: list[str]) -> NDArray[np.float64]:
        result = np.zeros((len(utterances), self.n_classes), dtype=float)
        neighbors_labels = self._get_neighbors(utterances)

        for instance in range(neighbors_labels.shape[0]):
            deltas = np.sum(neighbors_labels[instance], axis=0).astype(int)

            for label in range(self.n_classes):
                p_true = self._prior_prob_true[label] * self._cond_prob_true[label, deltas[label]]
                p_false = self._prior_prob_false[label] * self._cond_prob_false[label, deltas[label]]
                result[instance, label] = p_true / (p_true + p_false)

        return result

    def clear_cache(self) -> None:
        self.vector_index.delete()

    def dump(self, path: str) -> None:
        self.metadata = MLKnnScorerDumpMetadata(
            db_dir=self.db_dir,
            n_classes=self.n_classes,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

        arrays_to_save = ArrayToSave(
            prior_prob_true=self._prior_prob_true,
            prior_prob_false=self._prior_prob_false,
            cond_prob_true=self._cond_prob_true,
            cond_prob_false=self._cond_prob_false,
        )
        np.savez(dump_dir / self.arrays_filename, **arrays_to_save)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            self.metadata: MLKnnScorerDumpMetadata = json.load(file)
        self.n_classes = self.metadata["n_classes"]

        arrays: ArrayToSave = np.load(dump_dir / self.arrays_filename)

        self._prior_prob_true = arrays["prior_prob_true"]
        self._prior_prob_false = arrays["prior_prob_false"]
        self._cond_prob_true = arrays["cond_prob_true"]
        self._cond_prob_false = arrays["cond_prob_false"]

        vector_index_client = VectorIndexClient(
            device=self.device,
            db_dir=self.metadata["db_dir"],
            embedder_batch_size=self.metadata["batch_size"],
            embedder_max_length=self.metadata["max_length"],
        )
        self.vector_index = vector_index_client.get_index(self.model_name)
