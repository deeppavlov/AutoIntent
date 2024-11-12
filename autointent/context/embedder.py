import json
import logging
import shutil
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
        model_name: str | Path,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        if Path(model_name).exists():
            self.load(model_name)
        else:
            self.embedding_model = SentenceTransformer(str(model_name), device=device)

        self.logger = logging.getLogger(__name__)

    def clear_ram(self) -> None:
        self.logger.debug("deleting embedder %s", self.model_name)
        self.embedding_model.cpu()
        del self.embedding_model

    def delete(self) -> None:
        self.clear_ram()
        shutil.rmtree(
            self.dump_dir, ignore_errors=True
        )  # TODO: `ignore_errors=True` is workaround for PermissionError: [WinError 5] Access is denied

    def dump(self, path: Path) -> None:
        self.dump_dir = path
        metadata = EmbedderDumpMetadata(
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        path.mkdir(parents=True, exist_ok=True)
        self.embedding_model.save(str(path / self.embedder_subdir))
        with (path / self.metadata_dict_name).open("w") as file:
            json.dump(metadata, file, indent=4)

    def load(self, path: Path | str) -> None:
        self.dump_dir = Path(path)
        path = Path(path)
        with (path / self.metadata_dict_name).open() as file:
            metadata: EmbedderDumpMetadata = json.load(file)
        self.batch_size = metadata["batch_size"]
        self.max_length = metadata["max_length"]

        self.embedding_model = SentenceTransformer(str(path / self.embedder_subdir), device=self.device)

    def embed(self, utterances: list[str]) -> npt.NDArray[np.float32]:
        self.logger.debug(
            "calculating embeddings with model %s, batch_size=%d, max_seq_length=%s, device=%s",
            self.model_name,
            self.batch_size,
            str(self.max_length),
            self.device,
        )
        if self.max_length is not None:
            self.embedding_model.max_seq_length = self.max_length
        return self.embedding_model.encode(utterances, convert_to_numpy=True, batch_size=self.batch_size)  # type: ignore[return-value]
