import numpy as np
from chromadb.types import Collection
from numpy.typing import NDArray

from autointent.modules.scoring.base import Context, ScoringModule


class MLKnnScorer(ScoringModule):
    _multilabel: bool
    _collection: Collection
    _n_classes: int
    _converter: callable
    _prior_prob_true: NDArray[np.float64]
    _prior_prob_false: NDArray[np.float64]
    _cond_prob_true: NDArray[np.float64]
    _cond_prob_false: NDArray[np.float64]

    def __init__(self, k: int, s: float = 1.0, ignore_first_neighbours: int = 0):
        self.k = k
        self.s = s
        self.ignore_first_neighbours = ignore_first_neighbours

    def fit(self, context: Context) -> None:
        self._multilabel = context.multilabel
        self._collection = context.get_best_collection()
        self._n_classes = context.n_classes
        self._converter = context.vector_index.metadata_as_labels

        dataset = self._collection.get(include=["embeddings", "metadatas"])
        features = np.array(dataset["embeddings"])
        labels = np.array(self._converter(dataset["metadatas"]))

        self._prior_prob_true, self._prior_prob_false = self._compute_prior(labels)
        self._cond_prob_true, self._cond_prob_false = self._compute_cond(features, labels)

    def _compute_prior(self, y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        prior_prob_true = (self.s + y.sum(axis=0)) / (self.s * 2 + y.shape[0])
        prior_prob_false = 1 - prior_prob_true
        return prior_prob_true, prior_prob_false

    def _compute_cond(
        self, x: NDArray[np.str_], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        c = np.zeros((self._n_classes, self.k + 1), dtype=int)
        cn = np.zeros((self._n_classes, self.k + 1), dtype=int)

        neighbors = self._get_neighbors(embeddings=x)

        for i in range(x.shape[0]):
            deltas = y[neighbors[i]].sum(axis=0).astype(int)
            for label in range(self._n_classes):
                if y[i, label] == 1:
                    c[label, deltas[label]] += 1
                else:
                    cn[label, deltas[label]] += 1

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_prob_true = (self.s + c) / (self.s * (self.k + 1) + c_sum[:, None])
        cond_prob_false = (self.s + cn) / (self.s * (self.k + 1) + cn_sum[:, None])

        return cond_prob_true, cond_prob_false

    def _get_neighbors(
            self,
            embeddings: NDArray[np.str_] | None = None,
            texts: list[str] | None = None,
        ) -> NDArray[np.int64]:
        query_res = self._collection.query(
            query_embeddings=embeddings,
            query_texts=texts,
            n_results=self.k + self.ignore_first_neighbours,
            include=["metadatas"]
        )
        return np.array(
            [self._converter(candidates[self.ignore_first_neighbours :]) for candidates in query_res["metadatas"]]
        )

    def predict(self, utterances: list[str]) -> NDArray[np.int64]:
        result = np.zeros((len(utterances), self._n_classes), dtype=int)
        neighbors = self._get_neighbors(texts=utterances)

        for i in range(len(utterances)):
            deltas = np.sum(np.array(neighbors[i]), axis=0).astype(int)

            for label in range(self._n_classes):
                p_true = self._prior_prob_true[label] * self._cond_prob_true[label, deltas[label]]
                p_false = self._prior_prob_false[label] * self._cond_prob_false[label, deltas[label]]
                result[i, label] = int(p_true >= p_false)

        return result

    def predict_proba(self, x: list[str]) -> NDArray[np.float64]:
        result = np.zeros((len(x), self._n_classes), dtype=float)
        np_x = np.array(x)
        neighbors = self._get_neighbors(np_x)

        for instance in range(np_x.shape[0]):
            deltas = np.sum(self.encoder.transform(np.array(neighbors[instance]).reshape(-1, 1)), axis=0).astype(int)

            for label in range(self._n_classes):
                p_true = self._prior_prob_true[label] * self._cond_prob_true[label, deltas[label]]
                p_false = self._prior_prob_false[label] * self._cond_prob_false[label, deltas[label]]
                result[instance, label] = p_true / (p_true + p_false)

        return result

    def clear_cache(self) -> None:
        model = self._collection._embedding_function._model  # noqa: SLF001
        model.to(device="cpu")
        del model
        self._collection = None
