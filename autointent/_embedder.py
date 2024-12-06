"""Module for managing embedding models using Sentence Transformers.

This module provides the `Embedder` class for managing, persisting, and loading
embedding models and calculating embeddings for input texts.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
from appdirs import user_cache_dir
from sentence_transformers import SentenceTransformer

from ._hash import Hasher


def get_embeddings_path(filename: str) -> Path:
    """
    Get the path to the embeddings file.

    This function constructs the full path to an embeddings file stored
    in a specific directory under the user's home directory. The embeddings
    file is named based on the provided filename, with the `.npy` extension
    added.

    :param filename: The name of the embeddings file (without extension).

    :return: The full path to the embeddings file.
    """
    return Path(user_cache_dir("autointent")) / "embeddings" / f"{filename}.npy"


class EmbedderDumpMetadata(TypedDict):
    """Metadata for saving and loading an Embedder instance."""

    batch_size: int
    """Batch size used for embedding calculations."""
    max_length: int | None
    """Maximum sequence length for the embedding model."""


class Embedder:
    """
    A wrapper for managing embedding models using Sentence Transformers.

    This class handles initialization, saving, loading, and clearing of
    embedding models, as well as calculating embeddings for input texts.
    """

    embedder_subdir: str = "sentence_transformers"
    metadata_dict_name: str = "metadata.json"

    def __init__(
        self,
        model_name: str | Path,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int | None = None,
        use_cache: bool = False,
    ) -> None:
        """
        Initialize the Embedder.

        :param model_name: Path to a local model directory or a Hugging Face model name.
        :param device: Device to run the model on (e.g., "cpu", "cuda").
        :param batch_size: Batch size for embedding calculations.
        :param max_length: Maximum sequence length for the embedding model.
        :param embedder_use_cache: Flag indicating whether to cache intermediate embeddings.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_cache = use_cache

        if Path(model_name).exists():
            self.load(model_name)
        else:
            self.embedding_model = SentenceTransformer(str(model_name), device=device)

        self.logger = logging.getLogger(__name__)

    def __hash__(self) -> int:
        """
        Compute a hash value for the Embedder.

        :returns: The hash value of the Embedder.
        """
        hasher = Hasher()
        for parameter in self.embedding_model.parameters():
            hasher.update(parameter.detach().cpu().numpy())
        hasher.update(self.max_length)
        return hasher.intdigest()

    def clear_ram(self) -> None:
        """Move the embedding model to CPU and delete it from memory."""
        self.logger.debug("Clearing embedder %s from memory", self.model_name)
        self.embedding_model.cpu()
        del self.embedding_model

    def delete(self) -> None:
        """Delete the embedding model and its associated directory."""
        self.clear_ram()
        shutil.rmtree(
            self.dump_dir,
            ignore_errors=True,
        )  # TODO: `ignore_errors=True` is workaround for PermissionError: [WinError 5] Access is denied

    def dump(self, path: Path) -> None:
        """
        Save the embedding model and metadata to disk.

        :param path: Path to the directory where the model will be saved.
        """
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
        """
        Load the embedding model and metadata from disk.

        :param path: Path to the directory where the model is stored.
        """
        self.dump_dir = Path(path)
        path = Path(path)
        with (path / self.metadata_dict_name).open() as file:
            metadata: EmbedderDumpMetadata = json.load(file)
        self.batch_size = metadata["batch_size"]
        self.max_length = metadata["max_length"]

        self.embedding_model = SentenceTransformer(str(path / self.embedder_subdir), device=self.device)

    def embed(self, utterances: list[str]) -> npt.NDArray[np.float32]:
        """
        Calculate embeddings for a list of utterances.

        :param utterances: List of input texts to calculate embeddings for.
        :return: A numpy array of embeddings.
        """
        if self.use_cache:
            hasher = Hasher()
            hasher.update(self)
            hasher.update(utterances)

            embeddings_path = get_embeddings_path(hasher.hexdigest())
            if embeddings_path.exists():
                return np.load(embeddings_path)  # type: ignore[no-any-return]

        self.logger.debug(
            "Calculating embeddings with model %s, batch_size=%d, max_seq_length=%s, device=%s",
            self.model_name,
            self.batch_size,
            str(self.max_length),
            self.device,
        )

        if self.max_length is not None:
            self.embedding_model.max_seq_length = self.max_length

        embeddings = self.embedding_model.encode(
            utterances,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )

        if self.use_cache:
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, embeddings)

        return embeddings  # type: ignore[return-value]
