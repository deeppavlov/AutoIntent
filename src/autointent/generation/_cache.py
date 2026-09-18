"""Helpers for caching structured outputs from LLM."""

from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from appdirs import user_cache_dir
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from autointent._dump_tools.unit_dumpers import PydanticModelDumper
from autointent._hash import Hasher

if TYPE_CHECKING:
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


def _remove_cache_entry(path: Path) -> None:
    """Remove a single on-disk cache entry.

    Each entry is a *directory* (``PydanticModelDumper.dump`` writes
    ``class_info.json`` + ``model_dump.json`` inside it), so eviction must use
    ``rmtree`` rather than ``unlink``. ``ignore_errors`` keeps a missing or
    partially removed entry from raising during cleanup.
    """
    shutil.rmtree(path, ignore_errors=True)


class PydanticDiskCache:
    """Key-agnostic memory + disk cache of pydantic models under ``user_cache_dir("autointent")/<subdir>``.

    Each entry is a directory written by :class:`PydanticModelDumper`. Subclasses (or callers)
    decide how keys are derived; this class only stores and loads by key.
    """

    def __init__(
        self, subdir: str, use_cache: bool = True, max_workers: int | None = None, batch_size: int = 100
    ) -> None:
        """Initialize the cache.

        Args:
            subdir: Directory name under the autointent user cache dir that holds this cache's entries.
            use_cache: Whether to use caching.
            max_workers: Maximum number of worker threads for parallel loading.
                        If None, uses min(32, os.cpu_count() + 4).
            batch_size: Number of cache files to process in each batch.
        """
        self.subdir = subdir
        self.use_cache = use_cache
        self._memory_cache: dict[str, BaseModel] = {}
        self.max_workers = max_workers
        self.batch_size = batch_size

        if self.use_cache:
            self._load_existing_cache()

    def _cache_dir(self) -> Path:
        """Directory holding this cache's entries (resolved at call time so tests can redirect it)."""
        return Path(user_cache_dir("autointent")) / self.subdir

    def _entry_path(self, key: str) -> Path:
        """On-disk directory of one entry."""
        return self._cache_dir() / key

    def _load_existing_cache(self) -> None:
        """Load all existing cache items from disk into memory."""
        cache_dir = self._cache_dir()

        if not cache_dir.exists():
            return

        # Each cache entry is a directory written by PydanticModelDumper, so
        # collect directories (filtering on is_file() here matched nothing and
        # silently disabled eager loading entirely).
        cache_files = [f for f in cache_dir.iterdir() if f.is_dir()]

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
            _remove_cache_entry(cache_file)
        else:
            return cache_file.name, cached_data

        return None

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
        cache_path = self._entry_path(cache_key)

        if cache_path.exists():
            try:
                cached_data = PydanticModelDumper.load(cache_path)

                if isinstance(cached_data, output_model):
                    logger.debug("Using cached structured output from disk for key: %s", cache_key)
                    # Add to memory cache for future access
                    self._memory_cache[cache_key] = cached_data
                    return cached_data

                logger.warning("Cached data type mismatch on disk, removing invalid cache")
                _remove_cache_entry(cache_path)
            except (ValidationError, ImportError) as e:
                logger.warning("Failed to load cached structured output from disk: %s", e)
                _remove_cache_entry(cache_path)

        return None

    def _save_to_disk(self, cache_key: str, result: BaseModel) -> None:
        """Save result to disk cache.

        Args:
            cache_key: The cache key to use.
            result: The result to cache.
        """
        cache_path = self._entry_path(cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        PydanticModelDumper.dump(result, cache_path, exists_ok=True)

    async def _load_from_disk_async(self, cache_key: str, output_model: type[T]) -> T | None:
        """Load cached result from disk asynchronously.

        Args:
            cache_key: The cache key to look up.
            output_model: Pydantic model class to parse the response into.

        Returns:
            Cached result if available and valid, None otherwise.
        """
        cache_path = self._entry_path(cache_key)

        if cache_path.exists():
            try:
                cached_data = await PydanticModelDumper.load_async(cache_path)

                if isinstance(cached_data, output_model):
                    logger.debug("Using cached structured output from disk for key: %s", cache_key)
                    # Add to memory cache for future access
                    self._memory_cache[cache_key] = cached_data
                    return cached_data

                logger.warning("Cached data type mismatch on disk, removing invalid cache")
                _remove_cache_entry(cache_path)
            except (ValidationError, ImportError) as e:
                logger.warning("Failed to load cached structured output from disk: %s", e)
                _remove_cache_entry(cache_path)

        return None

    async def _save_to_disk_async(self, cache_key: str, result: BaseModel) -> None:
        """Save result to disk cache asynchronously.

        Args:
            cache_key: The cache key to use.
            result: The result to cache.
        """
        cache_path = self._entry_path(cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        await PydanticModelDumper.dump_async(result, cache_path, exists_ok=True)

    def get_by_key(self, key: str, model_type: type[T]) -> T | None:
        """Return the cached model for ``key`` if present and of type ``model_type``."""
        if not self.use_cache:
            return None
        memory_result = self._check_memory_cache(key, model_type)
        if memory_result is not None:
            return memory_result
        return self._load_from_disk(key, model_type)

    def set_by_key(self, key: str, result: BaseModel) -> None:
        """Store ``result`` under ``key`` in memory and on disk."""
        if not self.use_cache:
            return
        self._memory_cache[key] = result
        self._save_to_disk(key, result)
        logger.debug("Cached %s for key: %s (memory and disk)", type(result).__name__, key)

    async def get_by_key_async(self, key: str, model_type: type[T]) -> T | None:
        """Async variant of :meth:`get_by_key`."""
        if not self.use_cache:
            return None
        memory_result = self._check_memory_cache(key, model_type)
        if memory_result is not None:
            return memory_result
        return await self._load_from_disk_async(key, model_type)

    async def set_by_key_async(self, key: str, result: BaseModel) -> None:
        """Async variant of :meth:`set_by_key`."""
        if not self.use_cache:
            return
        self._memory_cache[key] = result
        await self._save_to_disk_async(key, result)
        logger.debug("Cached %s for key: %s (memory and disk)", type(result).__name__, key)


class StructuredOutputCache(PydanticDiskCache):
    """Cache for structured output results, keyed by prompt, schema, generation params and model."""

    def __init__(self, use_cache: bool = True, max_workers: int | None = None, batch_size: int = 100) -> None:
        """Initialize the cache.

        Args:
            use_cache: Whether to use caching.
            max_workers: Maximum number of worker threads for parallel loading.
                        If None, uses min(32, os.cpu_count() + 4).
            batch_size: Number of cache files to process in each batch.
        """
        super().__init__("structured_outputs", use_cache=use_cache, max_workers=max_workers, batch_size=batch_size)

    def _get_cache_key(
        self,
        messages: list[Message],
        output_model: type[T],
        generation_params: dict[str, Any],
        model_name: str,
        base_url: str | None,
    ) -> str:
        """Generate a cache key for the given parameters.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            generation_params: Generation parameters.
            model_name: Name of the language model that will serve the request.
            base_url: Base URL of the API endpoint, or None for the default.

        Returns:
            Cache key as a hexadecimal string.
        """
        hasher = Hasher()
        hasher.update(json.dumps(messages))
        hasher.update(json.dumps(output_model.model_json_schema()))
        hasher.update(json.dumps(generation_params))
        hasher.update(model_name)
        hasher.update(base_url)
        return hasher.hexdigest()

    def get(
        self,
        messages: list[Message],
        output_model: type[T],
        generation_params: dict[str, Any],
        model_name: str,
        base_url: str | None,
    ) -> T | None:
        """Get cached result if available.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            generation_params: Generation parameters.
            model_name: Name of the language model that will serve the request.
            base_url: Base URL of the API endpoint, or None for the default.

        Returns:
            Cached result if available, None otherwise.
        """
        if not self.use_cache:
            return None
        return self.get_by_key(
            self._get_cache_key(messages, output_model, generation_params, model_name, base_url), output_model
        )

    def set(
        self,
        messages: list[Message],
        output_model: type[T],
        generation_params: dict[str, Any],
        result: T,
        model_name: str,
        base_url: str | None,
    ) -> None:
        """Cache the result.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            generation_params: Generation parameters.
            result: The result to cache.
            model_name: Name of the language model that will serve the request.
            base_url: Base URL of the API endpoint, or None for the default.
        """
        if not self.use_cache:
            return
        self.set_by_key(self._get_cache_key(messages, output_model, generation_params, model_name, base_url), result)

    async def get_async(
        self,
        messages: list[Message],
        output_model: type[T],
        generation_params: dict[str, Any],
        model_name: str,
        base_url: str | None,
    ) -> T | None:
        """Get cached result if available (async version).

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            generation_params: Generation parameters.
            model_name: Name of the language model that will serve the request.
            base_url: Base URL of the API endpoint, or None for the default.

        Returns:
            Cached result if available, None otherwise.
        """
        if not self.use_cache:
            return None
        return await self.get_by_key_async(
            self._get_cache_key(messages, output_model, generation_params, model_name, base_url), output_model
        )

    async def set_async(
        self,
        messages: list[Message],
        output_model: type[T],
        generation_params: dict[str, Any],
        result: T,
        model_name: str,
        base_url: str | None,
    ) -> None:
        """Cache the result (async version).

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            generation_params: Generation parameters.
            result: The result to cache.
            model_name: Name of the language model that will serve the request.
            base_url: Base URL of the API endpoint, or None for the default.
        """
        if not self.use_cache:
            return
        await self.set_by_key_async(
            self._get_cache_key(messages, output_model, generation_params, model_name, base_url), result
        )
