"""Module for managing embedding models with multiple backends.

This module provides the `Embedder` class for managing, persisting, and loading
embedding models and calculating embeddings for input texts using different backends.
"""

import importlib
import json
import logging
from pathlib import Path
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import torch

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

    _backend_path = "backend"
    _config_path = "config"
    _dump_dir: Path | None = None
    _backend: BaseEmbeddingBackend

    def __init__(self, embedder_config: EmbedderConfig) -> None:
        """Initialize the Embedder.

        Args:
            embedder_config: Config of embedder.
        """
        self.config = embedder_config.model_copy(deep=True)
        self._backend = self._init_backend()

    def _init_backend(self) -> BaseEmbeddingBackend:
        """Load and instantiate proper backend based on config type."""
        if isinstance(self.config, SentenceTransformerEmbeddingConfig):
            return SentenceTransformerEmbeddingBackend(self.config)
        if isinstance(self.config, OpenaiEmbeddingConfig):
            return OpenaiEmbeddingBackend(self.config)
        # Check if it's exactly the abstract base config (not a subclass)

        msg = f"Cannot instantiate abstract EmbedderConfig: {self.config.__repr__()}"
        raise TypeError(msg)

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

    def dump(self, path: Path) -> None:
        """Save the embedding model and metadata to disk.

        Args:
            path: Path to the directory where the model will be saved.
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save the backend
        self._backend.dump(path / self._backend_path)

        # Save the config with class info
        (path / self._config_path).mkdir(parents=True, exist_ok=True)
        class_info = {"name": self.config.__class__.__name__, "module": self.config.__class__.__module__}
        with (path / self._config_path / "class_info.json").open("w", encoding="utf-8") as file:
            json.dump(class_info, file, ensure_ascii=False, indent=4)
        with (path / self._config_path / "model_dump.json").open("w", encoding="utf-8") as file:
            json.dump(self.config.model_dump(), file, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, path: Path | str, override_config: EmbedderConfig | None = None) -> "Embedder":
        """Load the embedding model and metadata from disk.

        Args:
            path: Path to the directory where the model is stored.
            override_config: one can override presaved settings
        """
        path = Path(path)

        # Load config class information
        with (path / cls._config_path / "class_info.json").open("r", encoding="utf-8") as file:
            class_info = json.load(file)

        with (path / cls._config_path / "model_dump.json").open("r", encoding="utf-8") as file:
            content = json.load(file)

        # Dynamically load the config class
        model_type_module = importlib.import_module(class_info["module"])
        model_type: type[EmbedderConfig] = getattr(model_type_module, class_info["name"])
        config = model_type.model_validate(content)

        # Apply override config if provided
        if override_config is not None:
            # Merge override config with loaded config
            # Only override specific fields, preserving the original config type
            override_dict = override_config.model_dump(exclude_unset=True)
            config_dict = config.model_dump()
            config_dict.update(override_dict)
            config = model_type.model_validate(config_dict)

        # Create instance with the loaded/overridden config
        instance = cls(config)

        # Load the appropriate backend
        backend_path = path / cls._backend_path
        if isinstance(config, SentenceTransformerEmbeddingConfig):
            instance._backend = SentenceTransformerEmbeddingBackend.load(backend_path)  # noqa: SLF001
        elif isinstance(config, OpenaiEmbeddingConfig):
            instance._backend = OpenaiEmbeddingBackend.load(backend_path)  # noqa: SLF001
        else:
            msg = f"Cannot load abstract EmbedderConfig: {config.__repr__()}"
            raise TypeError(msg)

        return instance

    @overload
    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, *, return_tensors: Literal[True]
    ) -> torch.Tensor: ...

    @overload
    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, *, return_tensors: Literal[False] = False
    ) -> npt.NDArray[np.float32]: ...

    def embed(
        self,
        utterances: list[str],
        task_type: TaskTypeEnum | None = None,
        return_tensors: bool = False,
    ) -> npt.NDArray[np.float32] | torch.Tensor:
        """Calculate embeddings for a list of utterances.

        Args:
            utterances: List of input texts to calculate embeddings for.
            task_type: Type of task for which embeddings are calculated.
            return_tensors: If True, return a PyTorch tensor; otherwise, return a numpy array.

        Returns:
            A numpy array or PyTorch tensor of embeddings.
        """
        if return_tensors:
            return self._backend.embed(utterances, task_type, return_tensors=True)
        return self._backend.embed(utterances, task_type, return_tensors=False)

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
