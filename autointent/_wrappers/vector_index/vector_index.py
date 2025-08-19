"""Module for managing vector indexes using FAISS and embedding models.

This module provides the `VectorIndex` class to handle indexing, querying, and
management of embeddings for nearest neighbor search.
"""

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import assert_never

from autointent._wrappers import Embedder
from autointent.configs import EmbedderConfig, FaissConfig, OpenSearchConfig, TaskTypeEnum, VectorIndexConfig
from autointent.custom_types import Document, ListOfLabels

from .base_backend import BaseIndexBackend
from .faiss import FaissBackend
from .opensearch import OpenSearchBackend

logger = logging.getLogger(__name__)


class VectorIndex:
    """A class for managing a vector index and embedding models.

    This class allows adding, querying, and managing embeddings and their associated
    labels for efficient nearest neighbor search.
    """

    _index_path = "data"
    _embedder_path = "embedder"
    _config_path = "config"
    embedder: Embedder
    index: BaseIndexBackend

    def __init__(self, embedder_config: EmbedderConfig | Embedder, config: VectorIndexConfig) -> None:
        """Initialize the VectorIndex with an embedding model.

        Args:
            embedder_config: Configuration for the embedding model.
            config: settings for vector index.
            backend: vector index backend to use.
        """
        self.embedder = embedder_config if isinstance(embedder_config, Embedder) else Embedder(embedder_config)
        self.config = config

    def _init_index(self, vector_size: int) -> BaseIndexBackend:
        res: BaseIndexBackend
        if isinstance(self.config, FaissConfig):
            res = FaissBackend(config=self.config, vector_size=vector_size)
        elif isinstance(self.config, OpenSearchConfig):
            res = OpenSearchBackend(config=self.config, vector_size=vector_size)
        elif isinstance(self.config, VectorIndexConfig):
            msg = f"Passed abstract vector index config {self.config.__repr__()}"
            raise TypeError(msg)
        else:
            assert_never(self.config)
        return res

    def add(self, texts: list[str], labels: ListOfLabels) -> None:
        """Add texts and their corresponding labels to the index.

        Args:
            texts: List of input texts.
            labels: List of labels corresponding to the texts.
        """
        if len(texts) != len(labels):
            msg = f"Texts and labels lengths mismatch: {len(texts)=} !] {len(labels)=}"
            raise ValueError(msg)

        logger.debug("Adding embeddings to vector index %s", self.embedder.config.model_name)
        embeddings = self.embedder.embed(texts, TaskTypeEnum.passage)

        if not hasattr(self, "index"):
            self.index = self._init_index(vector_size=embeddings.shape[1])

        self.index.add(
            embeddings=embeddings, documents=[Document(text=t, label=i) for t, i in zip(texts, labels, strict=True)]
        )

    def clear_ram(self) -> None:
        """Clear the vector index from RAM."""
        logger.debug("Clearing vector index %s from RAM", self.embedder.config.model_name)
        self.embedder.clear_ram()
        self.index.clear_ram()

    def _search_by_text(self, texts: list[str], k: int) -> tuple[npt.NDArray[Any], list[list[Document]]]:
        """Search the index using text queries.

        Args:
            texts: List of input texts to search for.
            k: Number of nearest neighbors to return.

        Returns:
            List of search results for each query.
        """
        query_embedding: npt.NDArray[np.float64] = self.embedder.embed(texts, TaskTypeEnum.query)  # type: ignore[assignment]
        return self.index.query(query_embedding, k)

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
        return self.index.get_all_embeddings()

    def query(
        self,
        queries: list[str] | npt.NDArray[Any],
        k: int,
    ) -> tuple[list[list[float]], list[list[Document]]]:
        """Query the index to retrieve nearest neighbors.

        Args:
            queries: List of text queries or embedding vectors.
            k: Number of nearest neighbors to return for each query.

        Returns:
            A tuple containing:
                - `distances`: Corresponding distances for each neighbor.
                - `documents`: Corresponding documents for each neighbor.
        """
        func = self._search_by_text if isinstance(queries[0], str) else self.index.query
        cosine_similarities, documents = func(queries, k)  # type: ignore[arg-type]

        distances = [
            [float(cosine_sim) for cosine_sim in neighbors_similarities]
            for neighbors_similarities in cosine_similarities
        ]

        return distances, documents

    def dump(self, dir_path: Path) -> None:
        """Save the index and associated data to disk.

        Args:
            dir_path: Directory path where the data will be stored.
        """
        dir_path.mkdir(parents=True, exist_ok=True)
        self.dump_dir = dir_path

        self.index.dump(self.dump_dir / self._index_path)
        self.embedder.dump(self.dump_dir / self._embedder_path)

        (dir_path / self._config_path).mkdir(parents=True)
        class_info = {"name": self.config.__class__.__name__, "module": self.config.__class__.__module__}
        with (dir_path / self._config_path / "class_info.json").open("w", encoding="utf-8") as file:
            json.dump(class_info, file, ensure_ascii=False, indent=4)
        with (dir_path / self._config_path / "model_dump.json").open("w", encoding="utf-8") as file:
            json.dump(self.config.model_dump(), file, ensure_ascii=False, indent=4)

    @classmethod
    def load(
        cls,
        dir_path: Path,
        embedder_override_config: EmbedderConfig | None = None,
    ) -> "VectorIndex":
        """Load the index and associated data from disk.

        Args:
            dir_path: Directory path where the data is stored.
            embedder_override_config: override some settings like device and inference batch size
        """
        embedder = Embedder.load(dir_path / cls._embedder_path, override_config=embedder_override_config)

        with (dir_path / cls._config_path / "model_dump.json").open("r", encoding="utf-8") as file:
            content = json.load(file)

        with (dir_path / cls._config_path / "class_info.json").open("r", encoding="utf-8") as file:
            class_info = json.load(file)

        model_type_module = importlib.import_module(class_info["module"])
        model_type: type[VectorIndexConfig] = getattr(model_type_module, class_info["name"])
        config = model_type.model_validate(content)

        instance = cls(
            embedder_config=EmbedderConfig(),  # dummy embedder config
            config=config,
        )
        instance.embedder = embedder

        if isinstance(config, FaissConfig):
            instance.index = FaissBackend.load(dir_path / cls._index_path)
        elif isinstance(config, OpenSearchConfig):
            instance.index = OpenSearchBackend.load(dir_path / cls._index_path)
        elif isinstance(config, VectorIndexConfig):
            msg = f"Passed abstract config to vector index: {config.__repr__()}"
            raise TypeError(msg)
        else:
            assert_never(config)

        return instance
