from pathlib import Path

import numpy as np
import numpy.typing as npt

from sentence_transformers import SentenceTransformer


class Embedder:
    embedding_model: SentenceTransformer

    def __init__(self, device: str | None = None, model_name: str | None = None, model_path: Path | None = None, embedder_batch_size: int = 1) -> None:
        if (model_name and model_path) or (model_name is None and model_path is None):
            msg = "Embedder requires either model_name or model_path set"
            raise ValueError(msg)
        elif model_path:
            self.embedding_model = SentenceTransformer(str(model_path), device=device)
        elif model_name:
            self.model_name = model_name

        self.device = device
        self.embedder_batch_size = embedder_batch_size

    def delete(self) -> None:
        if hasattr(self, "embedding_model"):
            self.embedding_model.cpu()
            del self.embedding_model

    def dump(self, path: Path) -> None:
        self.embedding_model.save(str(path))

    def embed(self, utterances: list[str]) -> npt.NDArray[np.float32]:
        if not hasattr(self, "embedding_model"):
            self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
        return self.embedding_model.encode(utterances, convert_to_numpy=True, batch_size=self.embedder_batch_size)  # type: ignore[return-value]
