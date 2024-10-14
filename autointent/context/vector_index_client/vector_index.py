from typing import Any

import faiss
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer


class VectorIndex:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.labels: list[int] | list[list[int]] = []    # (n_samples,) or (n_samples, n_classes)

    def add(self, texts: list[str], labels: list[int] | list[list[int]]) -> None:
        self.texts = texts
        embeddings = self.embed(texts)
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)  # type: ignore # noqa: PGH003
        self.labels.extend(labels)

    def delete(self) -> None:
        if self.index:
            self.index.reset()
        self.labels = []

    def _search_by_text(self, texts: list[str], k: int) -> list[list[dict]]:
        query_embedding = self.embedding_model.encode(texts)
        return self._search_by_embedding(query_embedding, k)

    def _search_by_embedding(self, embedding: np.ndarray, k: int) -> list[list[dict]]:
        if embedding.ndim != 2:  # noqa: PLR2004
            msg = "`embedding` should be a 2D array of shape (n_queries, dim_size)"
            raise ValueError(msg)

        distances, indices = self.index.search(embedding, k)

        results = []
        for inds, dists in zip(indices, distances, strict=True):
            cur_res = []
            for ind, dist in zip(inds, dists, strict=True):
                cur_res.append({"id": ind, "distance": dist, "label": self.labels[ind]})
            results.append(cur_res)

        return results

    def get_all_embeddings(self) -> npt.NDArray[Any]:
        return self.index.reconstruct_n(0, self.index.ntotal)

    def get_all_labels(self) -> list[int] | list[list[int]]:
        return self.labels

    def query(
        self, queries: list[str] | list[npt.NDArray], k: int
    ) -> tuple[list[Any], list[list[float]], list[list[str]]]:
        """
        Arguments
        ---
        `queries`: list of string texts or list of numpy embeddings

        `k`: number of nearest neighbors to return for each query

        Return
        ---
        `labels`: list of integers (multiclass labels) or binary vectors (multilabel labels) of neighbors retrieved

        `distances`: corresponding distances between queries and neighbors retrieved

        `texts`: corresponding texts
        """
        func = self._search_by_text if isinstance(queries[0], str) else self._search_by_embedding
        all_results = func(queries, k)

        all_labels = [[self.labels[result["id"]] for result in results] for results in all_results]
        all_distances = [[result["distance"] for result in results] for results in all_results]
        all_texts = [[self.texts[result["id"]] for result in results] for results in all_results]

        return all_labels, all_distances, all_texts

    def embed(self, utterances: list[str]) -> npt.NDArray[np.float32]:
        return self.embedding_model.encode(utterances, convert_to_numpy=True)
