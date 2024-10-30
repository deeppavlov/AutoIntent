import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer


class EmbedderDumpMetadata(TypedDict):
    batch_size: int
    max_length: int | None


class Embedder:
    embedding_model: SentenceTransformer
    embedder_subdir: str = "sentence_transformers"
    metadata_dict_name: str = "metadata.json"

    def __init__(
        self,
        device: str,
        model_name: str | None = None,
        model_path: Path | None = None,
        batch_size: int = 1,
        max_length: int | None = None,
    ) -> None:
        if (model_name and model_path) or (model_name is None and model_path is None):
            msg = "Embedder requires either model_name or model_path set"
            raise ValueError(msg)
        if model_path:
            self.load(model_path, device)
        elif model_name:
            self.model_name = model_name

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def delete(self) -> None:
        if hasattr(self, "embedding_model"):
            self.embedding_model.cpu()
            del self.embedding_model

    def dump(self, path: Path) -> None:
        metadata = EmbedderDumpMetadata(
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        path.mkdir(parents=True, exist_ok=True)
        self.embedding_model.save(str(path / self.embedder_subdir))
        with (path / self.metadata_dict_name).open("w") as file:
            json.dump(metadata, file, indent=4)

    def load(self, path: Path, device: str) -> None:
        with (path / self.metadata_dict_name).open() as file:
            metadata: EmbedderDumpMetadata = json.load(file)
        self.batch_size = metadata["batch_size"]
        self.max_length = metadata["max_length"]

        self.embedding_model = SentenceTransformer(str(path / self.embedder_subdir), device=device)

    def embed(self, utterances: list[str]) -> npt.NDArray[np.float32]:
        if not hasattr(self, "embedding_model"):
            self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
        if self.max_length is not None:
            self.embedding_model.max_seq_length = self.max_length
        return self.embedding_model.encode(utterances, convert_to_numpy=True, batch_size=self.batch_size)  # type: ignore[return-value]
