from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import partial
from typing import TYPE_CHECKING, TypedDict, cast

import aiometer
import numpy as np
import numpy.typing as npt

from autointent._deps import require
from autointent._hash import Hasher
from autointent.configs._embedder import OpenaiEmbeddingConfig

from .base import BaseEmbeddingBackend

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    import openai
    from tiktoken import Encoding
    from typing_extensions import NotRequired


logger = logging.getLogger(__name__)

# Third-party embedding model ids (e.g. OpenRouter) are unknown to tiktoken; use a conservative encoding
# only for counting tokens when splitting batches.
_FALLBACK_TIKTOKEN_ENCODING = "cl100k_base"
_ERROR_DETAIL_LIMIT = 2000


def _compact_error_detail(value: object) -> str:
    """Render provider error details without letting huge bodies flood logs/results."""
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = repr(value)
    else:
        text = str(value)

    if len(text) <= _ERROR_DETAIL_LIMIT:
        return text
    return f"{text[:_ERROR_DETAIL_LIMIT]}... <truncated>"


def _openai_api_error_message(exc: BaseException, *, batch_size: int) -> str:
    """Build a RuntimeError message that preserves useful OpenAI/provider details."""
    details = [f"{exc.__class__.__name__}: {_compact_error_detail(exc)}"]

    for attr in ("status_code", "code", "type", "body"):
        value = getattr(exc, attr, None)
        if value is not None:
            details.append(f"{attr}={_compact_error_detail(value)}")

    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            details.append(f"response_status_code={status_code}")

        response_text = getattr(response, "text", None)
        if response_text:
            details.append(f"response_text={_compact_error_detail(response_text)}")

    return f"Error calling OpenAI API (batch_size={batch_size}): {'; '.join(details)}"


def _tiktoken_encoding_for_embedding_model(model_name: str) -> Encoding:
    """Resolve tiktoken encoding for batch sizing; fallback for unknown provider model ids."""
    require("openai")
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(
            "tiktoken has no mapping for embedding model %r; using %r for token counting "
            "(per-request batch limits are approximate).",
            model_name,
            _FALLBACK_TIKTOKEN_ENCODING,
        )
        return tiktoken.get_encoding(_FALLBACK_TIKTOKEN_ENCODING)


class EmbeddingsCreateKwargs(TypedDict):
    input: list[str]
    model: str
    dimensions: NotRequired[int]


class OpenaiEmbeddingBackend(BaseEmbeddingBackend):
    """OpenAI-based embedding backend implementation."""

    config: OpenaiEmbeddingConfig
    _client: openai.OpenAI | None = None
    _async_client: openai.AsyncOpenAI | None = None

    def __init__(self, config: OpenaiEmbeddingConfig) -> None:
        """Initialize the OpenAI backend.

        Args:
            config: Configuration for OpenAI embeddings.
        """
        require("openai")
        self.config = config
        self._event_loop: asyncio.AbstractEventLoop | None = None

        if config.max_concurrent is not None:
            self._init_event_loop()

    def _get_client(self) -> openai.OpenAI:
        """Get or create OpenAI client instance."""
        import openai

        if self._client is None:
            self._client = openai.OpenAI(
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                base_url=os.getenv("OPENAI_BASE_URL", None),
            )
        return self._client

    def _get_async_client(self) -> openai.AsyncOpenAI:
        """Get or create async OpenAI client instance."""
        import openai

        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                base_url=os.getenv("OPENAI_BASE_URL", None),
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
        hasher.update(str(self.config.max_tokens_in_batch))
        return hasher.intdigest()

    def _embed_uncached(self, utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:
        """Compute OpenAI embeddings without caching."""
        if len(utterances) == 0:
            msg = "Empty input"
            logger.error(msg)
            raise ValueError(msg)

        # Apply task-specific prompt
        if prompt:
            utterances = [f"{prompt} {utterance}" for utterance in utterances]

        logger.debug(
            "Calculating embeddings with OpenAI model %s, batch_size=%d, max_tokens_in_batch=%s, "
            "dimensions=%s, prompt=%s, max_concurrent=%s",
            self.config.model_name,
            self.config.batch_size,
            str(self.config.max_tokens_in_batch),
            str(self.config.dimensions),
            prompt,
            self.config.max_concurrent,
        )

        # Use async processing if max_concurrent is specified
        if self.config.max_concurrent is not None:
            return self._process_embeddings_async(utterances)
        return self._process_embeddings_sync(utterances)

    def _embedding_request_batches(self, utterances: list[str]) -> list[list[str]]:
        """Slice utterances into batches for each embeddings API call."""
        return _batch_strings_by_token_budget(
            utterances,
            model_name=self.config.model_name,
            max_strings_per_batch=self.config.batch_size,
            max_tokens_per_batch=self.config.max_tokens_in_batch,
        )

    def _process_embeddings_sync(self, utterances: list[str]) -> npt.NDArray[np.float32]:
        """Process embeddings synchronously."""
        client = self._get_client()
        all_embeddings = []

        for batch in self._embedding_request_batches(utterances):
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
                msg = _openai_api_error_message(e, batch_size=len(batch))
                logger.exception(msg)
                raise RuntimeError(msg) from e

        return np.array(all_embeddings, dtype=np.float32)

    def _process_embeddings_async(self, utterances: list[str]) -> npt.NDArray[np.float32]:
        """Process embeddings asynchronously using aiometer."""
        batches = self._embedding_request_batches(utterances)

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
            msg = _openai_api_error_message(e, batch_size=len(batch))
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
        return cast("npt.NDArray[np.float32]", similarity_matrix)

    def dump(self, path: Path) -> None:
        """Save the backend state to disk.

        Args:
            path: Path to the directory where the backend will be saved.
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save the configuration
        config_path = path / "config.json"
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(self.config.model_dump(mode="json"), file, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> OpenaiEmbeddingBackend:
        """Load the backend state from disk.

        Args:
            path: Path to the directory where the backend is stored.

        Returns:
            Loaded backend instance.
        """
        # Load configuration
        config_path = path / "config.json"
        with config_path.open("r", encoding="utf-8") as file:
            config_data = json.load(file)

        config = OpenaiEmbeddingConfig.model_validate(config_data)

        # Create instance
        return cls(config)


def _batch_strings_by_token_budget(
    texts: list[str],
    *,
    model_name: str,
    max_strings_per_batch: int,
    max_tokens_per_batch: int | None,
) -> list[list[str]]:
    """Split texts into API batches constrained by count and optional token sum."""
    if max_tokens_per_batch is None:
        return [texts[i : i + max_strings_per_batch] for i in range(0, len(texts), max_strings_per_batch)]

    encoding = _tiktoken_encoding_for_embedding_model(model_name)
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_tokens = 0

    for text in texts:
        tokens = len(encoding.encode(text))
        current_text = text

        if current_batch and (
            current_tokens + tokens > max_tokens_per_batch or len(current_batch) >= max_strings_per_batch
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        if tokens > max_tokens_per_batch:
            logger.warning(
                "Single utterance exceeds max_tokens_in_batch (%d); truncating for OpenAI embeddings.",
                max_tokens_per_batch,
            )
            truncated_ids = encoding.encode(text)[:max_tokens_per_batch]
            current_text = encoding.decode(truncated_ids)
            tokens = len(encoding.encode(current_text))

        current_batch.append(current_text)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    return batches
