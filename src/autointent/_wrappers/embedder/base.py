from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np

from ._sqlite_cache import get_embedding_cache, utterance_key

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    import torch

    from autointent.configs import EmbedderConfig, TaskTypeEnum


class BaseEmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    config: EmbedderConfig
    supports_training: bool = False
    supports_cache: bool = True

    @abstractmethod
    def __init__(self, config: EmbedderConfig) -> None:
        """Initialize the embedding backend with configuration."""
        ...

    @abstractmethod
    def clear_ram(self) -> None:
        """Clear the backend from RAM."""
        ...

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
        """Calculate embeddings for a list of utterances, using a per-utterance cache.

        Empty input, ``use_cache=False``, or a backend that opts out of caching
        (``supports_cache=False``) bypasses the cache and calls ``_embed_uncached``
        directly, preserving each backend's existing empty-input behavior.

        Args:
            utterances: List of input texts to calculate embeddings for.
            task_type: Type of task for which embeddings are calculated.
            return_tensors: If True, return a PyTorch tensor; otherwise, a numpy array.

        Returns:
            A numpy array or PyTorch tensor of embeddings.
        """
        prompt = self.config.get_prompt(task_type)
        if not utterances or not self.config.use_cache or not self.supports_cache:
            embeddings = self._embed_uncached(utterances, prompt)
        else:
            embeddings = self._embed_cached(utterances, prompt)
        if return_tensors:
            return self._to_tensor(embeddings)
        return embeddings

    def _embed_cached(self, utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:
        """Embed via the SQLite per-utterance cache: reuse hits, compute only misses."""
        cache = get_embedding_cache()
        model_hash = self.get_hash()
        keys = [utterance_key(model_hash, utterance, prompt) for utterance in utterances]
        unique_keys = list(dict.fromkeys(keys))
        cached = cache.get_many(model_hash, unique_keys)
        missing = [key for key in unique_keys if key not in cached]
        if missing:
            key_to_utterance: dict[str, str] = {}
            for utterance, key in zip(utterances, keys, strict=True):
                if key in cached or key in key_to_utterance:
                    continue
                key_to_utterance[key] = utterance
            missing_utterances = [key_to_utterance[key] for key in missing]
            computed = self._embed_uncached(missing_utterances, prompt)
            new_entries = {key: computed[index] for index, key in enumerate(missing)}
            cache.set_many(model_hash, new_entries)
            cached.update(new_entries)
        return cast("npt.NDArray[np.float32]", np.stack([cached[key] for key in keys]))

    @abstractmethod
    def _embed_uncached(self, utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:
        """Compute embeddings WITHOUT caching, returning a ``(N, dim)`` float32 array.

        The backend applies ``prompt`` in its own way (ST passes it to ``encode``;
        OpenAI/vLLM prepend it; HashingVectorizer ignores it). Each backend keeps its
        current empty-input behavior here (ST/OpenAI/vLLM raise; HV returns ``(0, dim)``).
        """
        ...

    def _to_tensor(self, embeddings: npt.NDArray[np.float32]) -> torch.Tensor:
        """Convert a numpy embedding matrix to a torch tensor (CPU by default)."""
        import torch

        return torch.from_numpy(embeddings)

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
    def load(cls, path: Path) -> BaseEmbeddingBackend:
        """Load the backend state from disk.

        Args:
            path: Path to the directory where the backend is stored.

        Returns:
            Loaded backend instance.
        """
        ...
