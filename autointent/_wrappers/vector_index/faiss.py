from typing import Any

from numpy._typing import NDArray

from autointent.configs import VectorIndexConfig

from .base_backend import BaseBackend, Document


class FaissBackend(BaseBackend):
    def __init__(self, config: VectorIndexConfig) -> None:
        try:
            import faiss
        except ImportError as e:
            msg = "Unable to create Faiss vector index. Install faiss-cpu python package first."
            raise RuntimeError(msg) from e

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
