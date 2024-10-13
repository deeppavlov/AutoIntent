import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from .data_handler import DataHandler


class Index(ABC):
    @abstractmethod
    def add(self, embeddings: npt.NDArray[Any], metadata: list[dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def delete(self) -> None:
        pass

    @abstractmethod
    def search_by_query(self, query: str, k: int = 5) -> list[dict]:
        pass

    @abstractmethod
    def search_by_embedding(self, embedding: npt.NDArray[Any], k: int = 5) -> list[dict]:
        pass

    @abstractmethod
    def get_metadata(self) -> list[dict[str, Any]] | list[str]:
        pass

    @abstractmethod
    def get_all_embeddings(self) -> npt.NDArray[Any]:
        pass


class FaissIndex(Index):
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
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

    def get_all_embeddings(self) ->  npt.NDArray[Any]:
        return self.index.reconstruct_n(0, self.index.ntotal)


class VectorIndex:
    def __init__(self, device: str, multilabel: bool, n_classes: int) -> None:
        self._logger = logging.getLogger(__name__)
        self.device = device
        self.multilabel = multilabel
        self.n_classes = n_classes
        self.indexes: dict[str, Index] = {}

    def create_index(self, model_name: str, data_handler: DataHandler) -> str:
        self._logger.info(f"Creating index for model: {model_name}")

        index = FaissIndex(model_name, self.device)

        embeddings = index.embedding_model.encode(data_handler.utterances_train)
        metadata = self.labels_as_metadata(data_handler.labels_train)

        index.add(embeddings, metadata)

        self.indexes[model_name] = index
        return model_name

    def search(self, model_name: str, query: str, k: int = 5) -> list[dict]:
        return self.indexes[model_name].search_by_query(query, k)

    def search_by_embedding(self, model_name: str, embedding: np.ndarray, k: int = 5) -> list[dict]:
        return self.indexes[model_name].search_by_embedding(embedding, k)

    def query(
        self, model_name: str, queries: list[str], k: int, converter: Callable[[list[list[dict[str, int]]]], list[Any]]
    ) -> tuple[list[Any], list[list[float]]]:
        index = self.indexes[model_name]
        all_results = []
        for query in queries:
            all_results.append(index.search_by_query(query, k))

        all_metadata = [[result["metadata"] for result in results] for results in all_results]
        all_distances = [[result["distance"] for result in results] for results in all_results]

        labels = [converter(candidates) for candidates in all_metadata]

        return labels, all_distances

    def delete_index(self, model_name: str) -> None:
        if model_name in self.indexes:
            self._logger.debug(f"Deleting index for model: {model_name}")
            self.indexes[model_name].delete()
            del self.indexes[model_name]

    def metadata_as_labels(self, metadata: list[dict]) -> list[list[int]] | list[int]:
        if self.multilabel:
            return _multilabel_metadata_as_labels(metadata, self.n_classes)
        return _multiclass_metadata_as_labels(metadata)

    def labels_as_metadata(self, labels: list[Any]) -> list[dict]:
        if self.multilabel:
            return _multilabel_labels_as_metadata(labels)
        return _multiclass_labels_as_metadata(labels)

    def get_index(self, model_name: str) -> Index:
        return self.indexes[model_name]


def _multiclass_labels_as_metadata(labels_list: list[int]) -> list[dict[str, Any]]:
    return [{"intent_id": lab} for lab in labels_list]


def _multilabel_labels_as_metadata(labels_list: list[list[int]]) -> list[dict[str, Any]]:
    return [{str(i): lab for i, lab in enumerate(labs)} for labs in labels_list]


def _multiclass_metadata_as_labels(metadata: list[dict[str, Any]]) -> list[int]:
    return [dct["intent_id"] for dct in metadata]


def _multilabel_metadata_as_labels(metadata: list[dict], n_classes: int) -> list[list[int]]:
    return [[dct[str(i)] for i in range(n_classes)] for dct in metadata]