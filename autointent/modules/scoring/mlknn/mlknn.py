import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from autointent import Context
from autointent.context.vector_index_client import VectorIndexClient
from autointent.modules.scoring.base import ScoringModule


class MLKnnScorerDumpMetadata(TypedDict):
    db_dir: str
    device: str
    n_classes: int
    model_name: str


class MLKnnScorer(ScoringModule):
    _multilabel: bool
    _n_classes: int
    _prior_prob_true: NDArray[np.float64]
    _prior_prob_false: NDArray[np.float64]
    _cond_prob_true: NDArray[np.float64]
    _cond_prob_false: NDArray[np.float64]
    metadata_dict_name: str = "metadata.json"
    arrays_filename: str = "probs.npz"
    name = "mlknn"

    def __init__(self, k: int, s: float = 1.0, ignore_first_neighbours: int = 0) -> None:
        self.k = k
        self.s = s
        self.ignore_first_neighbours = ignore_first_neighbours

    def fit(self, context: Context) -> None:
        self.vector_index = context.get_best_index()
        self._n_classes = context.n_classes

        self.metadata = MLKnnScorerDumpMetadata(
            db_dir=str(context.db_dir),
            device=context.device,
            n_classes=self._n_classes,
            model_name=self.vector_index.model_name,
        )

        self.features = self.vector_index.get_all_embeddings()
        self.labels = np.array(self.vector_index.get_all_labels())
        self._prior_prob_true, self._prior_prob_false = self._compute_prior(self.labels)
        self._cond_prob_true, self._cond_prob_false = self._compute_cond()

    def _compute_prior(self, y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        prior_prob_true = (self.s + y.sum(axis=0)) / (self.s * 2 + y.shape[0])
        prior_prob_false = 1 - prior_prob_true
        return prior_prob_true, prior_prob_false

    def _compute_cond(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        c = np.zeros((self._n_classes, self.k + 1), dtype=int)
        cn = np.zeros((self._n_classes, self.k + 1), dtype=int)

        neighbors_labels = self._get_neighbors(self.features)

        for i in range(self.features.shape[0]):
            deltas = np.sum(neighbors_labels[i], axis=0).astype(int)
            idx_helper = np.arange(self._n_classes)
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
        result = np.zeros((len(utterances), self._n_classes), dtype=float)
        neighbors_labels = self._get_neighbors(utterances)

        for instance in range(neighbors_labels.shape[0]):
            deltas = np.sum(neighbors_labels[instance], axis=0).astype(int)

            for label in range(self._n_classes):
                p_true = self._prior_prob_true[label] * self._cond_prob_true[label, deltas[label]]
                p_false = self._prior_prob_false[label] * self._cond_prob_false[label, deltas[label]]
                result[instance, label] = p_true / (p_true + p_false)

        return result

    def clear_cache(self) -> None:
        self.vector_index.delete()

    def dump(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

        arrays_to_save = {
            "prior_prob_true": self._prior_prob_true,
            "prior_prob_false": self._prior_prob_false,
            "cond_prob_true": self._cond_prob_true,
            "cond_prob_false": self._cond_prob_false,
        }
        np.savez(dump_dir / self.arrays_filename, **arrays_to_save)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            self.metadata = json.load(file)
        self._n_classes = self.metadata["n_classes"]

        arrays = np.load(dump_dir / self.arrays_filename)

        self._prior_prob_true = arrays["prior_prob_true"]
        self._prior_prob_false = arrays["prior_prob_false"]
        self._cond_prob_true = arrays["cond_prob_true"]
        self._cond_prob_false = arrays["cond_prob_false"]

        vector_index_client = VectorIndexClient(device=self.metadata["device"], db_dir=self.metadata["db_dir"])
        self.vector_index = vector_index_client.get_index(self.metadata["model_name"])
