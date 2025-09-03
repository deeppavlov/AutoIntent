import asyncio
import logging
from functools import partial
from pathlib import Path
from typing import Literal, TypedDict, cast, overload

import aiometer
import numpy as np
import numpy.typing as npt
import openai
import torch
from typing_extensions import NotRequired

from autointent._hash import Hasher
from autointent.configs import TaskTypeEnum
from autointent.configs._embedder import OpenaiEmbeddingConfig

from .base import BaseEmbeddingBackend
from .utils import get_embeddings_path

logger = logging.getLogger(__name__)


class EmbeddingsCreateKwargs(TypedDict):
    input: list[str]
    model: str
    dimensions: NotRequired[int]


class OpenaiEmbeddingBackend(BaseEmbeddingBackend):
    """OpenAI-based embedding backend implementation."""

    def __init__(self, config: OpenaiEmbeddingConfig) -> None:
        """Initialize the OpenAI backend.

        Args:
            config: Configuration for OpenAI embeddings.
        """
        self.config = config
        self._client: openai.OpenAI | None = None
        self._async_client: openai.AsyncOpenAI | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

        if config.max_concurrent is not None:
            self._init_event_loop()

    def _get_client(self) -> openai.OpenAI:
        """Get or create OpenAI client instance."""
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def _get_async_client(self) -> openai.AsyncOpenAI:
        """Get or create async OpenAI client instance."""
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._async_client

    def _init_event_loop(self) -> None:
        """Initialize the asyncio event loop for async processing."""
        if self.config.max_concurrent is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            self._event_loop = loop

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
            task_type: Type of task for which embeddings are calculated.
            return_tensors: If True, return a PyTorch tensor; otherwise, return a numpy array.

        Returns:
            A numpy array or PyTorch tensor of embeddings.
        """
        if len(utterances) == 0:
            msg = "Empty input"
            logger.error(msg)
            raise ValueError(msg)

        # Apply task-specific prompt
        prompt = self.config.get_prompt(task_type)
        if prompt:
            utterances = [f"{prompt} {utterance}" for utterance in utterances]

        if self.config.use_cache:
            logger.debug("Using cached embeddings for %s", self.config.model_name)
            hasher = Hasher()
            hasher.update(self.get_hash())
            hasher.update(utterances)
            if prompt:
                hasher.update(prompt)

            embeddings_path = get_embeddings_path(hasher.hexdigest())
            if embeddings_path.exists():
                logger.debug("loading embeddings from %s", str(embeddings_path))
                embeddings_np = cast(npt.NDArray[np.float32], np.load(embeddings_path))
                if return_tensors:
                    return torch.from_numpy(embeddings_np)
                return embeddings_np

        logger.debug(
            "Calculating embeddings with OpenAI model %s, batch_size=%d, dimensions=%s, prompt=%s, max_concurrent=%s",
            self.config.model_name,
            self.config.batch_size,
            str(self.config.dimensions),
            prompt,
            self.config.max_concurrent,
        )

        # Use async processing if max_concurrent is specified
        if self.config.max_concurrent is not None:
            embeddings_np = self._process_embeddings_async(utterances)
        else:
            embeddings_np = self._process_embeddings_sync(utterances)

        if self.config.use_cache:
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, embeddings_np)

        if return_tensors:
            return torch.from_numpy(embeddings_np)
        return embeddings_np

    def _process_embeddings_sync(self, utterances: list[str]) -> np.ndarray:
        """Process embeddings synchronously."""
        client = self._get_client()
        all_embeddings = []

        # Process in batches
        for i in range(0, len(utterances), self.config.batch_size):
            batch = utterances[i : i + self.config.batch_size]

            # Prepare API call parameters
            kwargs: EmbeddingsCreateKwargs = {
                "input": batch,
                "model": self.config.model_name,
            }
            if self.config.dimensions is not None:
                kwargs["dimensions"] = self.config.dimensions

            try:
                response = client.embeddings.create(**kwargs)
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                msg = "Error calling OpenAI API"
                logger.exception(msg)
                raise RuntimeError(msg) from e

        return np.array(all_embeddings, dtype=np.float32)

    def _process_embeddings_async(self, utterances: list[str]) -> np.ndarray:
        """Process embeddings asynchronously using aiometer."""
        # Create batches
        batches = []
        for i in range(0, len(utterances), self.config.batch_size):
            batch = utterances[i : i + self.config.batch_size]
            batches.append(batch)

        # Create async tasks
        tasks = [partial(self._process_batch_async, batch) for batch in batches]

        # Run tasks with aiometer
        task = aiometer.run_all(
            tasks,
            max_at_once=self.config.max_concurrent,
            max_per_second=self.config.max_per_second,
        )
        if self._event_loop is None:
            msg = "Event loop is not initialized"
            raise RuntimeError(msg)
        batch_results = self._event_loop.run_until_complete(task)

        # Flatten results
        all_embeddings = [e for batch_embeddings in batch_results for e in batch_embeddings]

        return np.array(all_embeddings, dtype=np.float32)

    async def _process_batch_async(self, batch: list[str]) -> list[list[float]]:
        """Process a single batch asynchronously."""
        client = self._get_async_client()

        # Prepare API call parameters
        kwargs: EmbeddingsCreateKwargs = {
            "input": batch,
            "model": self.config.model_name,
        }
        if self.config.dimensions is not None:
            kwargs["dimensions"] = self.config.dimensions

        try:
            response = await client.embeddings.create(**kwargs)
            return [data.embedding for data in response.data]
        except Exception as e:
            msg = f"Error calling OpenAI API for batch: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

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
        return cast(npt.NDArray[np.float32], similarity_matrix)

    def dump(self, path: Path) -> None:
        """Save the backend state to disk.

        Args:
            path: Path to the directory where the backend will be saved.
        """
        import json

        path.mkdir(parents=True, exist_ok=True)

        # Save the configuration
        config_path = path / "config.json"
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(self.config.model_dump(mode="json"), file, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "OpenaiEmbeddingBackend":
        """Load the backend state from disk.

        Args:
            path: Path to the directory where the backend is stored.

        Returns:
            Loaded backend instance.
        """
        import json

        from autointent.configs._embedder import OpenaiEmbeddingConfig

        # Load configuration
        config_path = path / "config.json"
        with config_path.open("r", encoding="utf-8") as file:
            config_data = json.load(file)

        config = OpenaiEmbeddingConfig.model_validate(config_data)

        # Create instance
        return cls(config)
