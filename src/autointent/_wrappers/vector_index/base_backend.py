from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    from typing_extensions import Self

    from autointent.configs import VectorIndexConfig
    from autointent.custom_types import Document


MANIFEST_FILENAME = "remote_manifest.json"
"""Marker file inside a dump directory: the dump references remote cluster state.

Any backend whose ``dump()`` leaves data in an external engine writes this file
(``{"engine": ..., "index": ..., "dump_id": ...}``) so that dump-deletion tooling
can clean up the referenced cluster index (see ``remote_dumps.remove_module_dump``).
"""


class BaseIndexBackend(ABC):
    @abstractmethod
    def __init__(self, config: VectorIndexConfig, vector_size: int) -> None: ...

    @abstractmethod
    def add(self, embeddings: npt.NDArray[Any], documents: list[Document]) -> None:
        """Add documents and their embeddings to the index.

        The first ``add()`` call on a fresh instance must start from empty index
        contents (fit-replaces semantics); subsequent calls append. Backends with
        durable shared state (e.g. OpenSearch) satisfy this by resetting the
        remote index on their first write; purely local backends (Faiss) satisfy
        it by construction.
        """

    @abstractmethod
    def clear_ram(self) -> None:
        """Release local (in-process) resources held by the backend.

        Must not destroy durable index state: after ``clear_ram()``, a backend
        restored via ``load()`` from a previous ``dump()`` must still serve queries.
        """

    @abstractmethod
    def reset(self) -> None:
        """Drop all indexed documents, including durable state shared across instances."""

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
