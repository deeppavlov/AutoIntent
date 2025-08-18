import json
import platform
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from numpy._typing import NDArray
from typing_extensions import Self

from autointent.configs import FaissConfig
from autointent.custom_types import Document

from .base_backend import BaseIndexBackend


@contextmanager
def _limit_openmp_threads_on_darwin() -> Generator[None, None, None]:
    """Context manager that limits OpenMP threads to 1 on Darwin (macOS) platform.

    This helps avoid potential threading issues with FAISS on macOS by ensuring
    single-threaded execution during query operations.
    """
    if platform.system() != "Darwin":
        # Not on macOS, no need to limit threads
        yield
        return

    from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

    # Limit OpenMP threads to 1 on macOS
    with threadpool_limits(limits=1, user_api="openmp"):
        yield


class FaissBackend(BaseIndexBackend):
    _documents_filaname = "documents.json"
    _config_filename = "config.json"
    _index_filename = "index.bin"
    _vector_size_filename = "vector_size.txt"

    def __init__(self, config: FaissConfig, vector_size: int) -> None:
        try:
            import faiss

            self._faiss = faiss
        except ImportError as e:
            msg = "Unable to create Faiss vector index. Install faiss-cpu python package first."
            raise RuntimeError(msg) from e

        self.vector_size = vector_size
        self.config = config
        self._index = faiss.IndexFlatIP(self.vector_size)
        self._documents: list[Document] = []

    def clear_ram(self) -> None:
        self._index.reset()

    def get_all_embeddings(self) -> NDArray[Any]:
        return self._index.reconstruct_n(0, self._index.ntotal)  # type: ignore[no-any-return]

    def add(self, embeddings: NDArray[Any], documents: list[Document]) -> None:
        self._index.add(embeddings)

        self._documents.extend(documents)

    def query(self, embedding: NDArray[Any], k: int) -> tuple[NDArray[Any], list[list[Document]]]:
        with _limit_openmp_threads_on_darwin():
            cosine_similarities, indices = self._index.search(embedding, k)
            documents = [[self._documents[i] for i in neighbors_ids] for neighbors_ids in indices]
            return cosine_similarities, documents

    def dump(self, path: Path) -> None:
        data = [d.model_dump(mode="json") for d in self._documents]
        path.mkdir(parents=True)
        with (path / self._documents_filaname).open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        with (path / self._config_filename).open("w", encoding="utf-8") as file:
            json.dump(self.config.model_dump(mode="json"), file, indent=4, ensure_ascii=False)
        with (path / self._vector_size_filename).open("w", encoding="utf-8") as file:
            file.write(str(self.vector_size))
        self._faiss.write_index(self._index, str(path / self._index_filename))

    @classmethod
    def load(cls, path: Path) -> Self:
        with (path / cls._config_filename).open("r", encoding="utf-8") as file:
            config = FaissConfig.model_validate_json(file.read())

        with (path / cls._vector_size_filename).open("r", encoding="utf-8") as file:
            vector_size = int(file.read())

        instance = cls(config=config, vector_size=vector_size)

        with (path / cls._documents_filaname).open("r", encoding="utf-8") as file:
            docs_data = json.load(file)

        instance._documents = [Document.model_validate(d) for d in docs_data]  # noqa: SLF001

        instance._index = instance._faiss.read_index(str(path / cls._index_filename))  # noqa: SLF001

        return instance
