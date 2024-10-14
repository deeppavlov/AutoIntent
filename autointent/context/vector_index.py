import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from .data_handler import DataHandler


class VectorIndex:
    def __init__(self, model_name: str, device: str, converter: Callable) -> None:
        self.model_name = model_name
        self.device = device
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.metadata: list[dict] = []
        self.converter = converter

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)  # type: ignore # noqa: PGH003
        self.metadata.extend(metadata)

    def delete(self) -> None:
        if self.index:
            self.index.reset()
        self.metadata = []

    def search_by_query(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.embedding_model.encode([query])
        return self.search_by_embedding(query_embedding, k)

    def search_by_embedding(self, embedding: np.ndarray, k: int = 5) -> list[dict]:
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)

        distances, indices = self.index.search(embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({"id": idx, "distance": distances[0][i], "metadata": self.metadata[idx]})

        return results

    def get_metadata(self) -> list[dict]:
        return self.metadata

    def get_all_embeddings(self) -> npt.NDArray[Any]:
        return self.index.reconstruct_n(0, self.index.ntotal)

    def get_all_labels(self) -> list[int] | list[list[int]]:
        return [self.converter(mtd) for mtd in self.metadata]

    def query(
        self, queries: list[str] | list[npt.NDArray], k: int
    ) -> tuple[list[Any], list[list[float]]]:
        if isinstance(queries[0], str):
            all_results = [self.search_by_query(text, k) for text in queries]
        else:
            all_results = [self.search_by_embedding(emb, k) for emb in queries]

        all_metadata = [[result["metadata"] for result in results] for results in all_results]
        all_distances = [[result["distance"] for result in results] for results in all_results]

        labels = [self.converter(candidates) for candidates in all_metadata] if self.converter else all_metadata

        return labels, all_distances

    def embed(self, utterances: list[str]) -> npt.NDArray[np.float32]:
        return self.embedding_model.encode(utterances, convert_to_numpy=True)


class VectorIndexClient:
    def __init__(self, device: str, multilabel: bool, n_classes: int) -> None:
        self._logger = logging.getLogger(__name__)
        self.device = device
        self.multilabel = multilabel
        self.n_classes = n_classes
        self.indexes: dict[str, VectorIndex] = {}
        self.model_name = None

    def set_best_embedder_name(self, model_name: str) -> None:
        if model_name not in self.indexes:
            msg = f"model {model_name} wasn't created before"
            self._logger.error(msg)
            raise ValueError(msg)

        self.model_name = model_name

    def create_index(self, model_name: str, data_handler: DataHandler) -> VectorIndex:
        self._logger.info("Creating index for model: %s", model_name)

        index = VectorIndex(model_name, self.device, self.build_converter())

        embeddings = index.embedding_model.encode(data_handler.utterances_train)
        metadata = self.labels_as_metadata(data_handler.labels_train)

        index.add(embeddings, metadata)

        self.indexes[model_name] = index

        return index

    def delete_index(self, model_name: str) -> None:
        if model_name in self.indexes:
            self._logger.debug("Deleting index for model: %s", model_name)
            self.indexes[model_name].delete()
            del self.indexes[model_name]

    def build_converter(self) -> Callable:
        if self.multilabel:
            return partial(_multilabel_metadata_as_labels, n_classes=self.n_classes)
        return _multiclass_metadata_as_labels

    def labels_as_metadata(self, labels: list[Any]) -> list[dict]:
        if self.multilabel:
            return _multilabel_labels_as_metadata(labels)
        return _multiclass_labels_as_metadata(labels)

    def get_index(self, model_name: str) -> VectorIndex:
        return self.indexes[model_name]


def _multiclass_labels_as_metadata(labels_list: list[int]) -> list[dict[str, Any]]:
    return [{"intent_id": lab} for lab in labels_list]


def _multilabel_labels_as_metadata(labels_list: list[list[int]]) -> list[dict[str, Any]]:
    return [{str(i): lab for i, lab in enumerate(labs)} for labs in labels_list]


def _multiclass_metadata_as_labels(metadata: list[dict[str, Any]]) -> list[int]:
    return [dct["intent_id"] for dct in metadata]


def _multilabel_metadata_as_labels(metadata: list[dict], n_classes: int) -> list[list[int]]:
    return [[dct[str(i)] for i in range(n_classes)] for dct in metadata]
