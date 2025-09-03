import logging
import time
from pathlib import Path
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import openai
import torch
from appdirs import user_cache_dir

from autointent._hash import Hasher
from autointent.configs import TaskTypeEnum
from autointent.configs._embedder import OpenaiEmbeddingConfig

from .base import BaseEmbeddingBackend

logger = logging.getLogger(__name__)


def _get_embeddings_path(filename: str) -> Path:
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


class OpenaiEmbeddingBackend(BaseEmbeddingBackend):
    """OpenAI-based embedding backend implementation."""

    def __init__(self, config: OpenaiEmbeddingConfig) -> None:
        """Initialize the OpenAI backend.

        Args:
            config: Configuration for OpenAI embeddings.
        """
        self.config = config
        self._client = None

    def _get_client(self) -> openai.OpenAI:
        """Get or create OpenAI client instance."""
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def clear_ram(self) -> None:
        """Clear the backend from RAM. For OpenAI, this is a no-op."""
        # OpenAI API doesn't store models in RAM, so nothing to clear

    def get_hash(self) -> int:
        """Compute a hash value for identifying embedding model."""
        hasher = Hasher()
        hasher.update(self.config.model_name)
        hasher.update(str(self.config.dimensions))
        return hasher.intdigest()

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
            task_type: Type of task for which embeddings are calculated (unused for OpenAI).
            return_tensors: If True, return a PyTorch tensor; otherwise, return a numpy array.

        Returns:
            A numpy array or PyTorch tensor of embeddings.
        """
        if len(utterances) == 0:
            msg = "Empty input"
            logger.error(msg)
            raise ValueError(msg)

        if self.config.use_cache:
            logger.debug("Using cached embeddings for %s", self.config.model_name)
            hasher = Hasher()
            hasher.update(self.get_hash())
            hasher.update(utterances)

            embeddings_path = _get_embeddings_path(hasher.hexdigest())
            if embeddings_path.exists():
                logger.debug("loading embeddings from %s", str(embeddings_path))
                embeddings_np = np.load(embeddings_path).astype(np.float32)
                if return_tensors:
                    return torch.from_numpy(embeddings_np)
                return embeddings_np

        client = self._get_client()

        logger.debug(
            "Calculating embeddings with OpenAI model %s, batch_size=%d, dimensions=%s",
            self.config.model_name,
            self.config.batch_size,
            str(self.config.dimensions),
        )

        all_embeddings = []

        # Process in batches
        for i in range(0, len(utterances), self.config.batch_size):
            batch = utterances[i : i + self.config.batch_size]

            # Prepare API call parameters
            kwargs = {
                "input": batch,
                "model": self.config.model_name,
            }
            if self.config.dimensions is not None:
                kwargs["dimensions"] = self.config.dimensions

            try:
                response = client.embeddings.create(**kwargs)
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

                # Add small delay to avoid rate limiting
                if i + self.config.batch_size < len(utterances):
                    time.sleep(0.1)

            except Exception as e:
                msg = "Error calling OpenAI API"
                logger.exception(msg)
                raise RuntimeError(msg) from e

        embeddings_np = np.array(all_embeddings, dtype=np.float32)

        if self.config.use_cache:
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, embeddings_np)

        if return_tensors:
            return torch.from_numpy(embeddings_np)
        return embeddings_np

    def similarity(
        self, embeddings1: npt.NDArray[np.float32], embeddings2: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Calculate cosine similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings (size n).
            embeddings2: Second set of embeddings (size m).

        Returns:
            A numpy array of similarities (size n x m).
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

        normalized1 = embeddings1 / norm1
        normalized2 = embeddings2 / norm2

        # Calculate cosine similarity
        similarity_matrix = np.dot(normalized1, normalized2.T)
        return similarity_matrix.astype(np.float32)
