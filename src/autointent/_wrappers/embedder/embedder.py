"""Module for managing embedding models with multiple backends.

This module provides the `Embedder` class for managing, persisting, and loading
embedding models and calculating embeddings for input texts using different backends.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import torch
from typing_extensions import assert_never

from autointent.configs import EmbedderFineTuningConfig, TaskTypeEnum
from autointent.configs._embedder import EmbedderConfig, OpenaiEmbeddingConfig, SentenceTransformerEmbeddingConfig
from autointent.custom_types import ListOfLabels

from .base import BaseEmbeddingBackend
from .openai import OpenaiEmbeddingBackend
from .sentence_transformers import SentenceTransformerEmbeddingBackend

logger = logging.getLogger(__name__)


class Embedder:
    """A wrapper for managing embedding models with multiple backends.

    This class handles initialization, saving, loading, and clearing of
    embedding models, as well as calculating embeddings for input texts.
    """

    _metadata_dict_name: str = "metadata.json"
    _weights_dir_name: str = "sentence_transformer"
    _dump_dir: Path | None = None
    _backend: BaseEmbeddingBackend

    def __init__(self, embedder_config: EmbedderConfig) -> None:
        """Initialize the Embedder.

        Args:
            embedder_config: Config of embedder.
        """
        self.config = embedder_config.model_copy(deep=True)
        self._backend = self._load_model()

    def _load_model(self) -> BaseEmbeddingBackend:
        """Load and instantiate proper backend based on config type."""
        if isinstance(self.config, SentenceTransformerEmbeddingConfig):
            return SentenceTransformerEmbeddingBackend(self.config)
        if isinstance(self.config, OpenaiEmbeddingConfig):
            return OpenaiEmbeddingBackend(self.config)
        if isinstance(self.config, EmbedderConfig):
            # Handle abstract base config case
            msg = f"Cannot instantiate abstract EmbedderConfig: {self.config.__repr__()}"
            raise TypeError(msg)
        assert_never(self.config)

    def _get_hash(self) -> int:
        """Compute a hash value for the Embedder.

        Returns:
            The hash value of the Embedder.
        """
        return self._backend.get_hash()

    def train(self, utterances: list[str], labels: ListOfLabels, config: EmbedderFineTuningConfig) -> None:
        """Train the embedding model (only supported for backends with training support).

        Args:
            utterances: List of training utterances.
            labels: List of labels corresponding to utterances.
            config: Fine-tuning configuration.
        """
        if not self._backend.supports_training:
            msg = f"Training is not supported for {self._backend.__class__.__name__} backend"
            raise NotImplementedError(msg)

        # Only SentenceTransformer backend currently implements training
        if isinstance(self._backend, SentenceTransformerEmbeddingBackend):
            self._backend.train(utterances, labels, config)
        else:
            msg = f"Training method not implemented for {self._backend.__class__.__name__}"
            raise NotImplementedError(msg)

    def clear_ram(self) -> None:
        """Move the embedding model to CPU and delete it from memory."""
        self._backend.clear_ram()

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
        path.mkdir(parents=True, exist_ok=True)

        # Delegate to backend
        self._backend.dump(path)

    @classmethod
    def load(cls, path: Path | str, override_config: EmbedderConfig | None = None) -> "Embedder":
        """Load the embedding model and metadata from disk.

        Args:
            path: Path to the directory where the model is stored.
            override_config: one can override presaved settings
        """
        path = Path(path)

        # Try to load from backend first (new format)
        config_path = path / "config.json"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as file:
                config_data = json.load(file)

            # Determine backend type from config data
            if "api_key" in config_data or "max_concurrent" in config_data:
                backend = OpenaiEmbeddingBackend.load(path)
            else:
                backend = SentenceTransformerEmbeddingBackend.load(path)

            # Apply override config if provided
            if override_config is not None:
                config_dict = {**backend.config.model_dump(), **override_config.model_dump(exclude_unset=True)}
                if isinstance(backend, OpenaiEmbeddingBackend):
                    backend.config = OpenaiEmbeddingConfig(**config_dict)
                else:
                    backend.config = SentenceTransformerEmbeddingConfig(**config_dict)

            # Create Embedder instance and set backend directly
            instance = cls.__new__(cls)
            instance.config = backend.config
            instance._backend = backend
            instance._dump_dir = path
            return instance

        # Fallback to legacy format (deprecated)
        else:
            with (path / cls._metadata_dict_name).open(encoding="utf-8") as file:
                config_data = json.load(file)

            # Determine the config type based on the saved data
            if "api_key" in config_data or "openai" in config_data.get("model_name", "").lower():
                config = OpenaiEmbeddingConfig.model_validate(config_data)
            else:
                config = SentenceTransformerEmbeddingConfig.model_validate(config_data)

            if override_config is not None:
                kwargs = {**config.model_dump(), **override_config.model_dump(exclude_unset=True)}
                if isinstance(config, SentenceTransformerEmbeddingConfig):
                    config = SentenceTransformerEmbeddingConfig(**kwargs)
                else:
                    config = OpenaiEmbeddingConfig(**kwargs)

            # Handle legacy max_length field
            max_length = config_data.get("max_length")
            if max_length is not None and isinstance(config, SentenceTransformerEmbeddingConfig):
                config.tokenizer_config.max_length = max_length

            return cls(config)

    @overload
    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, *, return_tensors: Literal[True]
    ) -> torch.Tensor: ...

    @overload
    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, *, return_tensors: Literal[False] = False
    ) -> npt.NDArray[np.float32]: ...

    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, return_tensors: bool = False
    ) -> npt.NDArray[np.float32] | torch.Tensor:
        """Calculate embeddings for a list of utterances.

        Args:
            utterances: List of input texts to calculate embeddings for.
            task_type: Type of task for which embeddings are calculated.
            return_tensors: If True, return a PyTorch tensor; otherwise, return a numpy array.

        Returns:
            A numpy array or PyTorch tensor of embeddings.
        """
        return self._backend.embed(utterances, task_type, return_tensors=return_tensors)

    def similarity(
        self, embeddings1: npt.NDArray[np.float32], embeddings2: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Calculate similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings (size n).
            embeddings2: Second set of embeddings (size m).

        Returns:
            A numpy array of similarities (size n x m).
        """
        return self._backend.similarity(embeddings1, embeddings2)
