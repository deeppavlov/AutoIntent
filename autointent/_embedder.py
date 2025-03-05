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
import torch
from appdirs import user_cache_dir
from sentence_transformers import SentenceTransformer

from ._hash import Hasher
from .configs import EmbedderConfig, TaskTypeEnum


def get_embeddings_path(filename: str) -> Path:
    """Get the path to the embeddings file.

    This function constructs the full path to an embeddings file stored
    in a specific directory under the user's home directory. The embeddings
    file is named based on the provided filename, with the `.npy` extension
    added.

    Args:
        filename: The name of the embeddings file (without extension).

    Returns:
        The full path to the embeddings file.
    """
    return Path(user_cache_dir("autointent")) / "embeddings" / f"{filename}.npy"


class EmbedderDumpMetadata(TypedDict):
    """Metadata for saving and loading an Embedder instance."""

    model_name: str
    """Name of the hugging face model or a local path to sentence transformers dump."""
    device: str | None
    """Torch notation for CPU or CUDA."""
    batch_size: int
    """Batch size used for embedding calculations."""
    max_length: int | None
    """Maximum sequence length for the embedding model."""
    use_cache: bool
    """Whether to use embeddings caching."""


class Embedder:
    """A wrapper for managing embedding models using :py:class:`sentence_transformers.SentenceTransformer`.

    This class handles initialization, saving, loading, and clearing of
    embedding models, as well as calculating embeddings for input texts.
    """

    _metadata_dict_name: str = "metadata.json"
    _dump_dir: Path | None = None
    config: EmbedderConfig
    embedding_model: SentenceTransformer

    def __init__(self, embedder_config: EmbedderConfig) -> None:
        """Initialize the Embedder.

        Args:
            embedder_config: Config of embedder.
        """
        self.config = embedder_config

        self.embedding_model = SentenceTransformer(
            self.config.model_name, device=self.config.device, prompts=embedder_config.get_prompt_config()
        )

        self._logger = logging.getLogger(__name__)

    def __hash__(self) -> int:
        """Compute a hash value for the Embedder.

        Returns:
            The hash value of the Embedder.
        """
        hasher = Hasher()
        for parameter in self.embedding_model.parameters():
            hasher.update(parameter.detach().cpu().numpy())
        hasher.update(self.config.max_length)
        return hasher.intdigest()

    def clear_ram(self) -> None:
        """Move the embedding model to CPU and delete it from memory."""
        self._logger.debug("Clearing embedder %s from memory", self.config.model_name)
        self.embedding_model.cpu()
        del self.embedding_model
        torch.cuda.empty_cache()

    def delete(self) -> None:
        """Delete the embedding model and its associated directory."""
        self.clear_ram()
        if self._dump_dir is not None:
            shutil.rmtree(self._dump_dir)

    def dump(self, path: Path) -> None:
        """Save the embedding model and metadata to disk.

        Args:
            path: Path to the directory where the model will be saved.
        """
        self._dump_dir = path
        metadata = EmbedderDumpMetadata(
            model_name=str(self.config.model_name),
            device=self.config.device,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            use_cache=self.config.use_cache,
        )
        path.mkdir(parents=True, exist_ok=True)
        with (path / self._metadata_dict_name).open("w") as file:
            json.dump(metadata, file, indent=4)

    @classmethod
    def load(cls, path: Path | str, override_config: EmbedderConfig | None = None) -> "Embedder":
        """Load the embedding model and metadata from disk.

        Args:
            path: Path to the directory where the model is stored.
            override_config: one can override presaved settings
        """
        with (Path(path) / cls._metadata_dict_name).open() as file:
            metadata: EmbedderDumpMetadata = json.load(file)

        if override_config is not None:
            kwargs = {**metadata, **override_config.model_dump(exclude_unset=True)}
        else:
            kwargs = metadata  # type: ignore[assignment]

        return cls(EmbedderConfig(**kwargs))

    def embed(self, utterances: list[str], task_type: TaskTypeEnum | None = None) -> npt.NDArray[np.float32]:
        """Calculate embeddings for a list of utterances.

        Args:
            utterances: List of input texts to calculate embeddings for.
            task_type: Type of task for which embeddings are calculated.

        Returns:
            A numpy array of embeddings.
        """
        if self.config.use_cache:
            hasher = Hasher()
            hasher.update(self)
            hasher.update(utterances)

            embeddings_path = get_embeddings_path(hasher.hexdigest())
            if embeddings_path.exists():
                return np.load(embeddings_path)  # type: ignore[no-any-return]

        self._logger.debug(
            "Calculating embeddings with model %s, batch_size=%d, max_seq_length=%s, embedder_device=%s",
            self.config.model_name,
            self.config.batch_size,
            str(self.config.max_length),
            self.config.device,
        )

        if self.config.max_length is not None:
            self.embedding_model.max_seq_length = self.config.max_length

        embeddings = self.embedding_model.encode(
            utterances,
            convert_to_numpy=True,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            prompt_name=self.config.get_prompt_type(task_type),
        )

        if self.config.use_cache:
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, embeddings)

        return embeddings
