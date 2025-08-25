"""Helpers for caching structured outputs from LLM."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar

from appdirs import user_cache_dir
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from autointent._dump_tools.unit_dumpers import PydanticModelDumper
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

    def __init__(self, use_cache: bool = True, max_workers: int | None = None, batch_size: int = 100) -> None:
        """Initialize the cache.

        Args:
            use_cache: Whether to use caching.
            max_workers: Maximum number of worker threads for parallel loading.
                        If None, uses min(32, os.cpu_count() + 4).
            batch_size: Number of cache files to process in each batch.
        """
        self.use_cache = use_cache
        self._memory_cache: dict[str, BaseModel] = {}
        self.max_workers = max_workers
        self.batch_size = batch_size

        if self.use_cache:
            self._load_existing_cache()

    def _load_existing_cache(self) -> None:
        """Load all existing cache items from disk into memory."""
        cache_dir = Path(user_cache_dir("autointent")) / "structured_outputs"

        if not cache_dir.exists():
            return

        # Get all cache files to process
        cache_files = [f for f in cache_dir.iterdir() if f.is_file()]

        if not cache_files:
            return

        logger.debug("Loading %d cache files in batches of %d", len(cache_files), self.batch_size)

        # Process cache files in batches to avoid resource exhaustion
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self._load_cache_batch(executor, cache_files)

        logger.debug("Finished loading cache, %d items in memory", len(self._memory_cache))

    def _load_cache_batch(self, executor: ThreadPoolExecutor, cache_files: list[Path]) -> None:
        """Load cache files in batches using the provided executor.

        Args:
            executor: ThreadPoolExecutor to use for parallel processing.
            cache_files: List of cache files to load.
        """
        for i in range(0, len(cache_files), self.batch_size):
            batch = cache_files[i : i + self.batch_size]

            # Submit batch of cache loading tasks
            futures = [executor.submit(self._load_single_cache_file, cache_file) for cache_file in batch]

            # Process completed tasks in this batch
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    filename, cached_data = result
                    self._memory_cache[filename] = cached_data
                    logger.debug("Loaded cached item into memory: %s", filename)

    def _load_single_cache_file(self, cache_file: Path) -> tuple[str, BaseModel] | None:
        """Load a single cache file and return the result.

        Args:
            cache_file: Path to the cache file to load.

        Returns:
            Tuple of (filename, cached_data) if successful, None if failed.
        """
        try:
            cached_data = PydanticModelDumper.load(cache_file)
        except (ValidationError, ImportError) as e:
            logger.warning("Failed to load cached item %s: %s", cache_file.name, e)
            cache_file.unlink(missing_ok=True)
        else:
            return cache_file.name, cached_data

        return None

    def _get_cache_key(self, messages: list[Message], output_model: type[T], generation_params: dict[str, Any]) -> str:
        """Generate a cache key for the given parameters.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            generation_params: Generation parameters.

        Returns:
            Cache key as a hexadecimal string.
        """
        hasher = Hasher()
        hasher.update(json.dumps(messages))
        hasher.update(json.dumps(output_model.model_json_schema()))
        hasher.update(json.dumps(generation_params))
        return hasher.hexdigest()

    def _check_memory_cache(self, cache_key: str, output_model: type[T]) -> T | None:
        """Check if the result is available in memory cache.

        Args:
            cache_key: The cache key to look up.
            output_model: Pydantic model class to parse the response into.

        Returns:
            Cached result if available and valid, None otherwise.
        """
        if cache_key in self._memory_cache:
            cached_data = self._memory_cache[cache_key]
            if isinstance(cached_data, output_model):
                logger.debug("Using cached structured output from memory for key: %s", cache_key)
                return cached_data
            # Type mismatch, remove from memory cache
            del self._memory_cache[cache_key]
            logger.warning("Cached data type mismatch in memory, removing invalid cache")
        return None

    def _load_from_disk(self, cache_key: str, output_model: type[T]) -> T | None:
        """Load cached result from disk.

        Args:
            cache_key: The cache key to look up.
            output_model: Pydantic model class to parse the response into.

        Returns:
            Cached result if available and valid, None otherwise.
        """
        cache_path = _get_structured_output_cache_path(cache_key)

        if cache_path.exists():
            try:
                cached_data = PydanticModelDumper.load(cache_path)

                if isinstance(cached_data, output_model):
                    logger.debug("Using cached structured output from disk for key: %s", cache_key)
                    # Add to memory cache for future access
                    self._memory_cache[cache_key] = cached_data
                    return cached_data

                logger.warning("Cached data type mismatch on disk, removing invalid cache")
                cache_path.unlink()
            except (ValidationError, ImportError) as e:
                logger.warning("Failed to load cached structured output from disk: %s", e)
                cache_path.unlink(missing_ok=True)

        return None

    def _save_to_disk(self, cache_key: str, result: T) -> None:
        """Save result to disk cache.

        Args:
            cache_key: The cache key to use.
            result: The result to cache.
        """
        cache_path = _get_structured_output_cache_path(cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        PydanticModelDumper.dump(result, cache_path, exists_ok=True)

    def get(self, messages: list[Message], output_model: type[T], generation_params: dict[str, Any]) -> T | None:
        """Get cached result if available.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            generation_params: Generation parameters.

        Returns:
            Cached result if available, None otherwise.
        """
        if not self.use_cache:
            return None

        cache_key = self._get_cache_key(messages, output_model, generation_params)

        # First check in-memory cache
        memory_result = self._check_memory_cache(cache_key, output_model)
        if memory_result is not None:
            return memory_result

        # Fallback to disk cache
        return self._load_from_disk(cache_key, output_model)

    def set(self, messages: list[Message], output_model: type[T], generation_params: dict[str, Any], result: T) -> None:
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

        cache_key = self._get_cache_key(messages, output_model, generation_params)

        # Store in memory cache
        self._memory_cache[cache_key] = result

        # Store in disk cache
        self._save_to_disk(cache_key, result)
        logger.debug("Cached structured output for key: %s (memory and disk)", cache_key)

    async def _load_from_disk_async(self, cache_key: str, output_model: type[T]) -> T | None:
        """Load cached result from disk asynchronously.

        Args:
            cache_key: The cache key to look up.
            output_model: Pydantic model class to parse the response into.

        Returns:
            Cached result if available and valid, None otherwise.
        """
        cache_path = _get_structured_output_cache_path(cache_key)

        if cache_path.exists():
            try:
                cached_data = await PydanticModelDumper.load_async(cache_path)

                if isinstance(cached_data, output_model):
                    logger.debug("Using cached structured output from disk for key: %s", cache_key)
                    # Add to memory cache for future access
                    self._memory_cache[cache_key] = cached_data
                    return cached_data

                logger.warning("Cached data type mismatch on disk, removing invalid cache")
                cache_path.unlink()
            except (ValidationError, ImportError) as e:
                logger.warning("Failed to load cached structured output from disk: %s", e)
                cache_path.unlink(missing_ok=True)

        return None

    async def _save_to_disk_async(self, cache_key: str, result: T) -> None:
        """Save result to disk cache asynchronously.

        Args:
            cache_key: The cache key to use.
            result: The result to cache.
        """
        cache_path = _get_structured_output_cache_path(cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        await PydanticModelDumper.dump_async(result, cache_path, exists_ok=True)

    async def get_async(
        self, messages: list[Message], output_model: type[T], generation_params: dict[str, Any]
    ) -> T | None:
        """Get cached result if available (async version).

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            generation_params: Generation parameters.

        Returns:
            Cached result if available, None otherwise.
        """
        if not self.use_cache:
            return None

        cache_key = self._get_cache_key(messages, output_model, generation_params)

        # First check in-memory cache
        memory_result = self._check_memory_cache(cache_key, output_model)
        if memory_result is not None:
            return memory_result

        # Fallback to disk cache
        return await self._load_from_disk_async(cache_key, output_model)

    async def set_async(
        self, messages: list[Message], output_model: type[T], generation_params: dict[str, Any], result: T
    ) -> None:
        """Cache the result (async version).

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            backend: Backend to use for structured output.
            generation_params: Generation parameters.
            result: The result to cache.
        """
        if not self.use_cache:
            return

        cache_key = self._get_cache_key(messages, output_model, generation_params)

        # Store in memory cache
        self._memory_cache[cache_key] = result

        # Store in disk cache
        await self._save_to_disk_async(cache_key, result)
        logger.debug("Cached structured output for key: %s (memory and disk)", cache_key)
