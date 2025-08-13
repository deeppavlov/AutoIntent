import json
from pathlib import Path
from typing import Any

from numpy._typing import NDArray
from typing_extensions import Self

from autointent.configs import FaissConfig

from .base_backend import BaseBackend, Document


class FaissBackend(BaseBackend):
    _documents_filaname = "documents.json"
    _config_filename = "config.json"
    _index_filename = "index.bin"

    def __init__(self, config: FaissConfig) -> None:
        try:
            import faiss

            self._faiss = faiss
        except ImportError as e:
            msg = "Unable to create Faiss vector index. Install faiss-cpu python package first."
            raise RuntimeError(msg) from e

        self.config = config
        self._index = faiss.IndexFlatIP(config.vector_size)
        self._documents: list[Document] = []

    def clear_ram(self) -> None:
        self._index.reset()

    def get_all_embeddings(self) -> NDArray[Any]:
        return self._index.reconstruct_n(0, self._index.ntotal)  # type: ignore[no-any-return]

    def add(self, embeddings: NDArray[Any], documents: list[Document]) -> None:
        self._index.add(embeddings)

        self._documents.extend(documents)

    def query(self, embedding: NDArray[Any], k: int) -> tuple[NDArray[Any], list[list[Document]]]:
        cosine_similarities, indices = self._index.search(embedding, k)
        documents = [[self._documents[i] for i in neighbors_ids] for neighbors_ids in indices]
        return cosine_similarities, documents

    def dump(self, path: Path) -> None:
        data = [d.model_dump(mode="json") for d in self._documents]
        with (path / self._documents_filaname).open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        with (path / self._config_filename).open("w", encoding="utf-8") as file:
            json.dump(self.config, file, indent=4, ensure_ascii=False)
        self._faiss.write_index(self._index, str(path / self._index_filename))

    @classmethod
    def load(cls, path: Path) -> Self:
        with (path / cls._config_filename).open("r", encoding="utf-8") as file:
            config = FaissConfig.model_validate_json(file.read())

        instance = cls(config)

        with (path / cls._documents_filaname).open("r", encoding="utf-8") as file:
            docs_data = json.load(file)

        instance._documents = [Document.model_validate(d) for d in docs_data]  # noqa: SLF001

        instance._index = instance._faiss.read_index(str(path / cls._index_filename))  # noqa: SLF001

        return instance
