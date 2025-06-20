"""Helpers for caching structured outputs from LLM."""

import json
import logging
from pathlib import Path
from typing import Any, TypeVar

from appdirs import user_cache_dir
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from autointent._dump_tools import PydanticModelDumper
from autointent._hash import Hasher
from autointent.generation.chat_templates import Message

logger = logging.getLogger(__name__)

load_dotenv()

T = TypeVar("T", bound=BaseModel)
"""Type variable for Pydantic models used in structured output generation."""


def _get_structured_output_cache_path(dirname: str) -> Path:
    """Get the path to the structured output cache file.

    This function constructs the full path to a cache directory stored
    in a specific directory under the user's home directory. The cache
    directory is named based on the provided dirname.
    added.

    Args:
        dirname: The name of the cache file (without extension).

    Returns:
        The full path to the cache file.
    """
    return Path(user_cache_dir("autointent")) / "structured_outputs" / dirname


class StructuredOutputCache:
    """Cache for structured output results."""

    def __init__(self, use_cache: bool = True) -> None:
        """Initialize the cache.

        Args:
            use_cache: Whether to use caching.
        """
        self.use_cache = use_cache

    def _get_cache_key(
        self, messages: list[Message], output_model: type[T], backend: str, generation_params: dict[str, Any]
    ) -> str:
        """Generate a cache key for the given parameters.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            backend: Backend to use for structured output.
            generation_params: Generation parameters.

        Returns:
            Cache key as a hexadecimal string.
        """
        hasher = Hasher(strict=True)
        hasher.update(json.dumps(messages))
        hasher.update(json.dumps(output_model.model_json_schema()))
        hasher.update(backend)
        hasher.update(json.dumps(generation_params))
        return hasher.hexdigest()

    def get(
        self, messages: list[Message], output_model: type[T], backend: str, generation_params: dict[str, Any]
    ) -> T | None:
        """Get cached result if available.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            backend: Backend to use for structured output.
            generation_params: Generation parameters.

        Returns:
            Cached result if available, None otherwise.
        """
        if not self.use_cache:
            return None

        cache_key = self._get_cache_key(messages, output_model, backend, generation_params)
        cache_path = _get_structured_output_cache_path(cache_key)

        if cache_path.exists():
            try:
                cached_data = PydanticModelDumper.load(cache_path)

                if isinstance(cached_data, output_model):
                    logger.debug("Using cached structured output for key: %s", cache_key)
                    return cached_data

                logger.warning("Cached data type mismatch, removing invalid cache")
                cache_path.unlink()
            except (ValidationError, ImportError) as e:
                logger.warning("Failed to load cached structured output: %s", e)
                cache_path.unlink(missing_ok=True)

        return None

    def set(
        self, messages: list[Message], output_model: type[T], backend: str, generation_params: dict[str, Any], result: T
    ) -> None:
        """Cache the result.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            backend: Backend to use for structured output.
            generation_params: Generation parameters.
            result: The result to cache.
        """
        if not self.use_cache:
            return

        cache_key = self._get_cache_key(messages, output_model, backend, generation_params)
        cache_path = _get_structured_output_cache_path(cache_key)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        PydanticModelDumper.dump(result, cache_path, exists_ok=True)
        logger.debug("Cached structured output for key: %s", cache_key)
