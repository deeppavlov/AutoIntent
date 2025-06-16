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

from autointent._wrappers import Embedder
from autointent.configs import EmbedderConfig, TaskTypeEnum, TokenizerConfig
from autointent.custom_types import ListOfLabels


class VectorIndexMetadata(TypedDict):
    embedder_model_name: str
    embedder_device: str | None
    embedder_batch_size: int
    embedder_max_length: int | None
    embedder_use_cache: bool


class VectorIndexData(TypedDict):
    texts: list[str]
    labels: ListOfLabels


class VectorIndex:
    """A class for managing a vector index using FAISS and embedding models.

    This class allows adding, querying, and managing embeddings and their associated
    labels for efficient nearest neighbor search.
    """

    _data_file = "data.json"
    _meta_data_file = "metadata.json"

    def __init__(self, embedder_config: EmbedderConfig) -> None:
        """Initialize the VectorIndex with an embedding model.

        Args:
            embedder_config: Configuration for the embedding model.
        """
        self.embedder = Embedder(embedder_config)

        self.labels: ListOfLabels = []  # (n_samples,) or (n_samples, n_classes)
        self.texts: list[str] = []

        self._logger = logging.getLogger(__name__)

    def add(self, texts: list[str], labels: ListOfLabels) -> None:
        """Add texts and their corresponding labels to the index.

        Args:
            texts: List of input texts.
            labels: List of labels corresponding to the texts.
        """
        self._logger.debug("Adding embeddings to vector index %s", self.embedder.config.model_name)
        embeddings = self.embedder.embed(texts, TaskTypeEnum.passage)

        if not hasattr(self, "index"):
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.labels.extend(labels)  # type: ignore[arg-type]
        self.texts.extend(texts)

    def is_empty(self) -> bool:
        """Check if the index is empty.

        Returns:
            True if the index contains no embeddings, False otherwise.
        """
        return len(self.labels) == 0

    def delete(self) -> None:
        """Delete the vector index and all associated data from disk and memory."""
        self._logger.debug("Deleting vector index %s", self.embedder.config.model_name)
        self.embedder.delete()
        self.clear_ram()
        shutil.rmtree(self.dump_dir)

    def clear_ram(self) -> None:
        """Clear the vector index from RAM."""
        self._logger.debug("Clearing vector index %s from RAM", self.embedder.config.model_name)
        self.embedder.clear_ram()
        self.index.reset()
        self.labels = []
        self.texts = []

    def _search_by_text(self, texts: list[str], k: int) -> list[list[dict[str, Any]]]:
        """Search the index using text queries.

        Args:
            texts: List of input texts to search for.
            k: Number of nearest neighbors to return.

        Returns:
            List of search results for each query.
        """
        query_embedding: npt.NDArray[np.float64] = self.embedder.embed(texts, TaskTypeEnum.query)  # type: ignore[assignment]
        return self._search_by_embedding(query_embedding, k)

    def _search_by_embedding(self, embedding: npt.NDArray[Any], k: int) -> list[list[dict[str, Any]]]:
        """Search the index using embedding vectors.

        Args:
            embedding: 2D array of shape (n_queries, dim_size) representing query embeddings.
            k: Number of nearest neighbors to return.

        Returns:
            List of search results for each query.
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
        """Retrieve all embeddings stored in the index.

        Returns:
            Array of all embeddings.

        Raises:
            ValueError: If the index has not been created yet.
        """
        if not hasattr(self, "index"):
            msg = "Index is not created yet"
            raise ValueError(msg)
        return self.index.reconstruct_n(0, self.index.ntotal)  # type: ignore[no-any-return]

    def get_all_labels(self) -> ListOfLabels:
        """Retrieve all labels stored in the index.

        Returns:
            List of all labels.
        """
        return self.labels

    def query(
        self,
        queries: list[str] | npt.NDArray[np.float32],
        k: int,
    ) -> tuple[list[ListOfLabels], list[list[float]], list[list[str]]]:
        """Query the index to retrieve nearest neighbors.

        Args:
            queries: List of text queries or embedding vectors.
            k: Number of nearest neighbors to return for each query.

        Returns:
            A tuple containing:
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
        """Save the index and associated data to disk.

        Args:
            dir_path: Directory path where the data will be stored.
        """
        dir_path.mkdir(parents=True, exist_ok=True)
        self.dump_dir = dir_path

        data = VectorIndexData(texts=self.texts, labels=self.labels)
        with (self.dump_dir / self._data_file).open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

        metadata = VectorIndexMetadata(
            embedder_max_length=self.embedder.config.tokenizer_config.max_length,
            embedder_model_name=str(self.embedder.config.model_name),
            embedder_device=self.embedder.config.device,
            embedder_batch_size=self.embedder.config.batch_size,
            embedder_use_cache=self.embedder.config.use_cache,
        )

        with (self.dump_dir / self._meta_data_file).open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)

    @classmethod
    def load(
        cls,
        dir_path: Path,
        embedder_device: str | None = None,
        embedder_batch_size: int | None = None,
        embedder_use_cache: bool | None = None,
    ) -> "VectorIndex":
        """Load the index and associated data from disk.

        Args:
            dir_path: Directory path where the data is stored.
            embedder_device: Device for the embedding model.
            embedder_batch_size: Batch size for the embedding model.
            embedder_use_cache: Whether to use caching for the embedding model.
        """
        with (dir_path / cls._meta_data_file).open(encoding="utf-8") as file:
            metadata: VectorIndexMetadata = json.load(file)

        instance = cls(
            EmbedderConfig(
                model_name=metadata["embedder_model_name"],
                device=embedder_device or metadata["embedder_device"],
                batch_size=embedder_batch_size or metadata["embedder_batch_size"],
                tokenizer_config=TokenizerConfig(max_length=metadata["embedder_max_length"]),
                use_cache=embedder_use_cache or metadata["embedder_use_cache"],
            )
        )

        with (dir_path / cls._data_file).open(encoding="utf-8") as file:
            data: VectorIndexData = json.load(file)

        instance.add(**data)
        return instance
