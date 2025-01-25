"""Module for managing vector indexes using FAISS and embedding models.

This module provides the `VectorIndex` class to handle indexing, querying, and
management of embeddings for nearest neighbor search.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, TypedDict

import faiss
import numpy as np
import numpy.typing as npt

from autointent import Embedder
from autointent.custom_types import ListOfLabels


class VectorIndexMetadata(TypedDict):
    embedder_model_name: str
    embedder_device: str
    embedder_batch_size: int
    embedder_max_length: int | None
    embedder_use_cache: bool


class VectorIndexData(TypedDict):
    texts: list[str]
    labels: ListOfLabels


class VectorIndex:
    """
    A class for managing a vector index using FAISS and embedding models.

    This class allows adding, querying, and managing embeddings and their associated
    labels for efficient nearest neighbor search.
    """

    _data_file = "data.json"
    _meta_data_file = "metadata.json"

    def __init__(
        self,
        embedder_model_name: str,
        embedder_device: str,
        embedder_batch_size: int = 32,
        embedder_max_length: int | None = None,
        embedder_use_cache: bool = True,
    ) -> None:
        """
        Initialize the vector index.

        :param embedder_model_name: Name of the embedding model to use.
        :param embedder_device: Device for running the embedding model (e.g., "cpu", "cuda").
        :param embedder_batch_size: Batch size for the embedder.
        :param embedder_max_length: Maximum sequence length for the embedder.
        :param embedder_use_cache: Flag indicating whether to cache intermediate embeddings.
        """
        self.embedder = Embedder(
            model_name_or_path=embedder_model_name,
            batch_size=embedder_batch_size,
            device=embedder_device,
            max_length=embedder_max_length,
            use_cache=embedder_use_cache,
        )
        self.embedder_device = embedder_device

        self.labels: ListOfLabels = []  # (n_samples,) or (n_samples, n_classes)
        self.texts: list[str] = []

        self.logger = logging.getLogger(__name__)

    def add(self, texts: list[str], labels: ListOfLabels) -> None:
        """
        Add texts and their corresponding labels to the index.

        :param texts: List of input texts.
        :param labels: List of labels corresponding to the texts.
        """
        self.logger.debug("Adding embeddings to vector index %s", self.embedder.model_name)
        embeddings = self.embedder.embed(texts)

        if not hasattr(self, "index"):
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.labels.extend(labels)  # type: ignore[arg-type]
        self.texts.extend(texts)

    def is_empty(self) -> bool:
        """
        Check if the index is empty.

        :return: True if the index contains no embeddings, False otherwise.
        """
        return len(self.labels) == 0

    def delete(self) -> None:
        """Delete the vector index and all associated data from disk and memory."""
        self.logger.debug("Deleting vector index %s", self.embedder.model_name)
        self.embedder.delete()
        self.clear_ram()
        shutil.rmtree(self.dump_dir)

    def clear_ram(self) -> None:
        """Clear the vector index from RAM."""
        self.logger.debug("Clearing vector index %s from RAM", self.embedder.model_name)
        self.embedder.clear_ram()
        self.index.reset()
        self.labels = []
        self.texts = []

    def _search_by_text(self, texts: list[str], k: int) -> list[list[dict[str, Any]]]:
        """
        Search the index using text queries.

        :param texts: List of input texts to search for.
        :param k: Number of nearest neighbors to return.
        :return: List of search results for each query.
        """
        query_embedding: npt.NDArray[np.float64] = self.embedder.embed(texts)  # type: ignore[assignment]
        return self._search_by_embedding(query_embedding, k)

    def _search_by_embedding(self, embedding: npt.NDArray[Any], k: int) -> list[list[dict[str, Any]]]:
        """
        Search the index using embedding vectors.

        :param embedding: 2D array of shape (n_queries, dim_size) representing query embeddings.
        :param k: Number of nearest neighbors to return.
        :return: List of search results for each query.
        :raises ValueError: If the embedding array is not 2D.
        """
        if embedding.ndim != 2:  # noqa: PLR2004
            msg = "`embedding` should be a 2D array of shape (n_queries, dim_size)"
            raise ValueError(msg)

        cos_sim, indices = self.index.search(embedding, k)  # TODO add caching similar to Embedder.embed() caching
        distances = 1 - cos_sim

        results = []
        for inds, dists in zip(indices, distances, strict=True):
            cur_res = [
                {"id": ind, "distance": dist, "label": self.labels[ind]} for ind, dist in zip(inds, dists, strict=True)
            ]
            results.append(cur_res)

        return results

    def get_all_embeddings(self) -> npt.NDArray[Any]:
        """
        Retrieve all embeddings stored in the index.

        :return: Array of all embeddings.
        :raises ValueError: If the index has not been created yet.
        """
        if not hasattr(self, "index"):
            msg = "Index is not created yet"
            raise ValueError(msg)
        return self.index.reconstruct_n(0, self.index.ntotal)  # type: ignore[no-any-return]

    def get_all_labels(self) -> ListOfLabels:
        """
        Retrieve all labels stored in the index.

        :return: List of all labels.
        """
        return self.labels

    def query(
        self,
        queries: list[str] | npt.NDArray[np.float32],
        k: int,
    ) -> tuple[list[ListOfLabels], list[list[float]], list[list[str]]]:
        """
        Query the index to retrieve nearest neighbors.

        :param queries: List of text queries or embedding vectors.
        :param k: Number of nearest neighbors to return for each query.
        :return: A tuple containing:
                 - `labels`: List of retrieved labels for each query.
                 - `distances`: Corresponding distances for each neighbor.
                 - `texts`: Corresponding texts for each neighbor.
        """
        func = self._search_by_text if isinstance(queries[0], str) else self._search_by_embedding
        all_results = func(queries, k)  # type: ignore[arg-type]

        all_labels: list[ListOfLabels] = [[self.labels[result["id"]] for result in results] for results in all_results]
        all_distances = [[float(result["distance"]) for result in results] for results in all_results]
        all_texts: list[list[str]] = [[self.texts[result["id"]] for result in results] for results in all_results]

        return all_labels, all_distances, all_texts

    def dump(self, dir_path: Path) -> None:
        """
        Save the index and associated data to disk.

        :param dir_path: Directory path to save the data.
        """
        dir_path.mkdir(parents=True, exist_ok=True)
        self.dump_dir = dir_path

        data = VectorIndexData(texts=self.texts, labels=self.labels)
        with (self.dump_dir / self._data_file).open("w") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

        metadata = VectorIndexMetadata(
            embedder_max_length=self.embedder.max_length,
            embedder_model_name=str(self.embedder.model_name),
            embedder_device=self.embedder.device,
            embedder_batch_size=self.embedder.batch_size,
            embedder_use_cache=self.embedder.use_cache,
        )

        with (self.dump_dir / self._meta_data_file).open("w") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)

    @classmethod
    def load(
        cls,
        dir_path: Path,
        embedder_device: str | None = None,
        embedder_batch_size: int | None = None,
        embedder_use_cache: bool | None = None,
    ) -> "VectorIndex":
        """
        Load the index and associated data from disk.

        :param dir_path: Directory path where the data is stored.
        """
        with (dir_path / cls._meta_data_file).open() as file:
            metadata: VectorIndexMetadata = json.load(file)

        instance = cls(
            embedder_model_name=metadata["embedder_model_name"],
            embedder_device=embedder_device or metadata["embedder_device"],
            embedder_batch_size=embedder_batch_size or metadata["embedder_batch_size"],
            embedder_max_length=metadata["embedder_max_length"],
            embedder_use_cache=embedder_use_cache or metadata["embedder_use_cache"],
        )

        with (dir_path / cls._data_file).open() as file:
            data: VectorIndexData = json.load(file)

        instance.add(**data)
        return instance
