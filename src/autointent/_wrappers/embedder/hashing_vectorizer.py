"""HashingVectorizer-based embedding backend for lightweight testing."""

import json
import logging
from pathlib import Path
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from autointent._hash import Hasher
from autointent.configs import TaskTypeEnum
from autointent.configs._embedder import HashingVectorizerEmbeddingConfig

from .base import BaseEmbeddingBackend

logger = logging.getLogger(__name__)


class HashingVectorizerEmbeddingBackend(BaseEmbeddingBackend):
    """HashingVectorizer-based embedding backend implementation.

    This backend uses sklearn's HashingVectorizer for fast, stateless text vectorization.
    Ideal for testing as it requires no model downloads and is very fast.
    """

    supports_training: bool = False

    def __init__(self, config: HashingVectorizerEmbeddingConfig) -> None:
        """Initialize the HashingVectorizer backend.

        Args:
            config: Configuration for HashingVectorizer embeddings.
        """
        self.config = config
        self._vectorizer = HashingVectorizer(
            n_features=config.n_features,
            ngram_range=config.ngram_range,
            analyzer=config.analyzer,
            lowercase=config.lowercase,
            norm=config.norm,
            binary=config.binary,
            dtype=getattr(np, config.dtype),
        )

    def clear_ram(self) -> None:
        """Clear the backend from RAM.

        HashingVectorizer is stateless, so this is a no-op.
        """

    def get_hash(self) -> int:
        """Compute a hash value for the backend.

        Returns:
            The hash value of the backend.
        """
        hasher = Hasher()
        # Hash all relevant config parameters
        hasher.update(self.config.n_features)
        hasher.update(self.config.ngram_range)
        hasher.update(self.config.analyzer)
        hasher.update(self.config.lowercase)
        hasher.update(self.config.norm if self.config.norm is not None else "None")
        hasher.update(self.config.binary)
        hasher.update(self.config.dtype)
        return hasher.hexdigest()

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
        task_type: TaskTypeEnum | None = None,  # noqa: ARG002
        return_tensors: bool = False,
    ) -> npt.NDArray[np.float32] | torch.Tensor:
        """Calculate embeddings for a list of utterances.

        Args:
            utterances: List of input texts to calculate embeddings for.
            task_type: Type of task for which embeddings are calculated (ignored for HashingVectorizer).
            return_tensors: If True, return a PyTorch tensor; otherwise, return a numpy array.

        Returns:
            A numpy array or PyTorch tensor of embeddings.
        """
        # Transform texts to sparse matrix, then convert to dense
        embeddings_sparse = self._vectorizer.transform(utterances)
        embeddings = embeddings_sparse.toarray().astype(np.float32)

        if return_tensors:
            return torch.from_numpy(embeddings)
        return embeddings

    def similarity(
        self, embeddings1: npt.NDArray[np.float32], embeddings2: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Calculate cosine similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings with shape (n_samples, n_features).
            embeddings2: Second set of embeddings with shape (m_samples, n_features).

        Returns:
            Similarity matrix with shape (n_samples, m_samples).
        """
        return cosine_similarity(embeddings1, embeddings2).astype(np.float32)

    def dump(self, path: Path) -> None:
        """Save the backend state to disk.

        Args:
            path: Directory path where the backend should be saved.
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save a metadata file indicating this is a HashingVectorizer backend
        metadata = {
            "backend_type": "hashing_vectorizer",
            "config": self.config.model_dump(),
        }

        metadata_path = path / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.debug("Saved HashingVectorizer backend to %s", path)

    @classmethod
    def load(cls, path: Path) -> "HashingVectorizerEmbeddingBackend":
        """Load the backend from disk.

        Args:
            path: Directory path where the backend is stored.

        Returns:
            Loaded HashingVectorizerEmbeddingBackend instance.
        """
        metadata_path = path / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        config = HashingVectorizerEmbeddingConfig.model_validate(metadata["config"])
        instance = cls(config)

        logger.debug("Loaded HashingVectorizer backend from %s", path)
        return instance

    def train(self, utterances: list[str], labels: list[int], config) -> None:  # noqa: ANN001
        """Train the backend.

        HashingVectorizer is stateless and doesn't support training.

        Args:
            utterances: Training utterances.
            labels: Training labels.
            config: Training configuration.

        Raises:
            NotImplementedError: HashingVectorizer doesn't support training.
        """
        msg = "HashingVectorizer backend does not support training"
        raise NotImplementedError(msg)
