from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import torch

from autointent.configs import EmbedderConfig, TaskTypeEnum


class BaseEmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    supports_training: bool = False

    @abstractmethod
    def __init__(self, config: EmbedderConfig) -> None:
        """Initialize the embedding backend with configuration."""
        ...

    @abstractmethod
    def clear_ram(self) -> None:
        """Clear the backend from RAM."""
        ...

    @overload
    @abstractmethod
    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, *, return_tensors: Literal[True]
    ) -> torch.Tensor: ...

    @overload
    @abstractmethod
    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, *, return_tensors: Literal[False] = False
    ) -> npt.NDArray[np.float32]: ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def get_hash(self) -> int:
        """Compute a hash value for the backend configuration and model state.

        Returns:
            The hash value of the backend.
        """
        ...

    @abstractmethod
    def dump(self, path: Path) -> None:
        """Save the backend state to disk.

        Args:
            path: Path to the directory where the backend will be saved.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseEmbeddingBackend":
        """Load the backend state from disk.

        Args:
            path: Path to the directory where the backend is stored.

        Returns:
            Loaded backend instance.
        """
        ...
