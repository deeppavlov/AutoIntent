from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy.typing as npt
from typing_extensions import Self

from autointent.configs import VectorIndexConfig
from autointent.custom_types import Document


class BaseIndexBackend(ABC):
    @abstractmethod
    def __init__(self, config: VectorIndexConfig, vector_size: int) -> None: ...

    @abstractmethod
    def add(self, embeddings: npt.NDArray[Any], documents: list[Document]) -> None: ...

    @abstractmethod
    def clear_ram(self) -> None: ...

    @abstractmethod
    def query(self, embedding: npt.NDArray[Any], k: int) -> tuple[npt.NDArray[Any], list[list[Document]]]:
        """Query by embedding.

        Args:
            embedding: 2D numpy array of shape (n_queries, vector_size)
            k: number of nearest neighbors to return

        Return:
            cosine_similarities: 2D numpy array of shape (n_queries, k)
            documents: corresponding list of documents
        """

    @staticmethod
    def _validate_embeddings(embedding: npt.NDArray[Any]) -> None:
        """Check embeddings compatibility for search.

        Args:
            embedding: 2D numpy array of shape (n_queries, vector_size)

        Return:
            cosine_similarities: 2D numpy array of shape (n_queries, k)
            documents: corresponding list of documents
        """
        if embedding.ndim != 2:  # noqa: PLR2004
            msg = "`embedding` should be a 2D array of shape (n_queries, vector_size)"
            raise ValueError(msg)

    @abstractmethod
    def get_all_embeddings(self) -> npt.NDArray[Any]: ...

    @abstractmethod
    def dump(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self: ...
