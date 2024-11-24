"""Module for managing vector indexes using FAISS and embedding models.

This module provides the `VectorIndex` class to handle indexing, querying, and
management of embeddings for nearest neighbor search.
"""

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt

from autointent.context import Embedder
from autointent.custom_types import LabelType


class VectorIndex:
    """
    A class for managing a vector index using FAISS and embedding models.

    This class allows adding, querying, and managing embeddings and their associated
    labels for efficient nearest neighbor search.
    """

    def __init__(
        self, model_name: str, device: str, embedder_batch_size: int = 32, embedder_max_length: int | None = None
    ) -> None:
        """
        Initialize the vector index.

        :param model_name: Name of the embedding model to use.
        :param device: Device for running the embedding model (e.g., "cpu", "cuda").
        :param embedder_batch_size: Batch size for the embedder.
        :param embedder_max_length: Maximum sequence length for the embedder.
        """
        self.model_name = model_name
        self.embedder = Embedder(
            model_name=model_name,
            batch_size=embedder_batch_size,
            device=device,
            max_length=embedder_max_length,
        )
        self.device = device

        self.labels: list[LabelType] = []  # (n_samples,) or (n_samples, n_classes)
        self.texts: list[str] = []

        self.logger = logging.getLogger(__name__)

    def add(self, texts: list[str], labels: list[LabelType]) -> None:
        """
        Add texts and their corresponding labels to the index.

        :param texts: List of input texts.
        :param labels: List of labels corresponding to the texts.
        """
        self.logger.debug("Adding embeddings to vector index %s", self.model_name)
        embeddings = self.embedder.embed(texts)

        if not hasattr(self, "index"):
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.labels.extend(labels)
        self.texts.extend(texts)

    def is_empty(self) -> bool:
        """
        Check if the index is empty.

        :return: True if the index contains no embeddings, False otherwise.
        """
        return len(self.labels) == 0

    def delete(self) -> None:
        """Delete the vector index and all associated data from disk and memory."""
        self.logger.debug("Deleting vector index %s", self.model_name)
        self.embedder.delete()
        self.clear_ram()
        (self.dump_dir / "index.faiss").unlink()
        (self.dump_dir / "texts.json").unlink()
        (self.dump_dir / "labels.json").unlink()

    def clear_ram(self) -> None:
        """Clear the vector index from RAM."""
        self.logger.debug("Clearing vector index %s from RAM", self.model_name)
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

        cos_sim, indices = self.index.search(embedding, k)
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

    def get_all_labels(self) -> list[LabelType]:
        """
        Retrieve all labels stored in the index.

        :return: List of all labels.
        """
        return self.labels

    def query(
        self, queries: list[str] | npt.NDArray[np.float32], k: int
    ) -> tuple[list[list[LabelType]], list[list[float]], list[list[str]]]:
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

        all_labels = [[self.labels[result["id"]] for result in results] for results in all_results]
        all_distances = [[result["distance"] for result in results] for results in all_results]
        all_texts = [[self.texts[result["id"]] for result in results] for results in all_results]

        return all_labels, all_distances, all_texts

    def dump(self, dir_path: Path) -> None:
        """
        Save the index and associated data to disk.

        :param dir_path: Directory path to save the data.
        """
        dir_path.mkdir(parents=True, exist_ok=True)
        self.dump_dir = dir_path
        faiss.write_index(self.index, str(self.dump_dir / "index.faiss"))
        self.embedder.dump(self.dump_dir / "embedding_model")
        with (self.dump_dir / "texts.json").open("w") as file:
            json.dump(self.texts, file, indent=4, ensure_ascii=False)
        with (self.dump_dir / "labels.json").open("w") as file:
            json.dump(self.labels, file, indent=4, ensure_ascii=False)

    def load(self, dir_path: Path) -> None:
        """
        Load the index and associated data from disk.

        :param dir_path: Directory path where the data is stored.
        """
        self.dump_dir = Path(dir_path)
        self.index = faiss.read_index(str(dir_path / "index.faiss"))
        self.embedder = Embedder(model_name=dir_path / "embedding_model", device=self.device)
        with (dir_path / "texts.json").open() as file:
            self.texts = json.load(file)
        with (dir_path / "labels.json").open() as file:
            self.labels = json.load(file)
