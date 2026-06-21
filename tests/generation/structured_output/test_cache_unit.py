"""Unit tests for StructuredOutputCache memory/disk/async behavior (no LLM)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from autointent.generation._cache import StructuredOutputCache, _get_structured_output_cache_path
from autointent.generation.chat_templates import Role

if TYPE_CHECKING:
    from pathlib import Path

    from autointent.generation.chat_templates import Message


class CacheModel(BaseModel):
    name: str
    value: int


class OtherModel(BaseModel):
    text: str


MESSAGES: list[Message] = [{"role": Role.USER, "content": "hi"}]
PARAMS: dict[str, Any] = {"temperature": 0.0}


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the structured-output disk cache to a fresh tmp dir each test."""
    monkeypatch.setattr("autointent.generation._cache.user_cache_dir", lambda *_: str(tmp_path))


def test_set_get_memory_roundtrip() -> None:
    cache = StructuredOutputCache(use_cache=True)
    result = CacheModel(name="a", value=1)
    cache.set(MESSAGES, CacheModel, PARAMS, result)
    assert cache.get(MESSAGES, CacheModel, PARAMS) == result


def test_disabled_cache_is_noop() -> None:
    cache = StructuredOutputCache(use_cache=False)
    cache.set(MESSAGES, CacheModel, PARAMS, CacheModel(name="a", value=1))
    assert cache.get(MESSAGES, CacheModel, PARAMS) is None


def test_get_misses_for_unknown_key() -> None:
    cache = StructuredOutputCache(use_cache=True)
    assert cache.get(MESSAGES, CacheModel, PARAMS) is None


def test_get_loads_from_disk_in_fresh_instance() -> None:
    """A second instance has empty memory and must read the entry back from disk."""
    StructuredOutputCache(use_cache=True).set(MESSAGES, CacheModel, PARAMS, CacheModel(name="x", value=9))

    fresh = StructuredOutputCache(use_cache=True)
    fresh._memory_cache.clear()  # force the disk path even if eager load changes
    loaded = fresh.get(MESSAGES, CacheModel, PARAMS)
    assert isinstance(loaded, CacheModel)
    assert loaded.value == 9
    # disk hit populates the memory cache for next time
    assert fresh._memory_cache


def test_memory_type_mismatch_evicts() -> None:
    cache = StructuredOutputCache(use_cache=True)
    key = cache._get_cache_key(MESSAGES, CacheModel, PARAMS)
    cache._memory_cache[key] = OtherModel(text="wrong")
    assert cache._check_memory_cache(key, CacheModel) is None
    assert key not in cache._memory_cache


@pytest.mark.asyncio
async def test_async_set_get_roundtrip() -> None:
    cache = StructuredOutputCache(use_cache=True)
    result = CacheModel(name="async", value=7)
    await cache.set_async(MESSAGES, CacheModel, PARAMS, result)
    assert await cache.get_async(MESSAGES, CacheModel, PARAMS) == result


@pytest.mark.asyncio
async def test_async_get_loads_from_disk() -> None:
    await StructuredOutputCache(use_cache=True).set_async(MESSAGES, CacheModel, PARAMS, CacheModel(name="d", value=3))

    fresh = StructuredOutputCache(use_cache=True)
    fresh._memory_cache.clear()
    loaded = await fresh.get_async(MESSAGES, CacheModel, PARAMS)
    assert isinstance(loaded, CacheModel)
    assert loaded.value == 3


@pytest.mark.asyncio
async def test_async_disabled_cache_is_noop() -> None:
    cache = StructuredOutputCache(use_cache=False)
    await cache.set_async(MESSAGES, CacheModel, PARAMS, CacheModel(name="a", value=1))
    assert await cache.get_async(MESSAGES, CacheModel, PARAMS) is None


# --- Regression tests for the on-disk-cache bugs (#326 eager load, #327 eviction) ---
# Disk entries are directories (PydanticModelDumper writes class_info.json +
# model_dump.json), so eager load must collect directories and eviction must
# rmtree rather than unlink.


def test_eager_load_populates_memory_from_disk() -> None:
    """A fresh instance eagerly batch-loads existing on-disk entries into memory (#326)."""
    StructuredOutputCache(use_cache=True).set(MESSAGES, CacheModel, PARAMS, CacheModel(name="x", value=9))

    fresh = StructuredOutputCache(use_cache=True)
    key = fresh._get_cache_key(MESSAGES, CacheModel, PARAMS)

    # populated at construction by the eager load, before any get() call
    assert key in fresh._memory_cache
    assert isinstance(fresh._memory_cache[key], CacheModel)


def test_eager_load_removes_corrupted_entry() -> None:
    """A cache directory whose payload fails to load is skipped and cleaned up, not raised."""
    entry = _get_structured_output_cache_path("corrupted-entry")
    entry.mkdir(parents=True)
    (entry / "class_info.json").write_text(json.dumps({"name": CacheModel.__name__, "module": CacheModel.__module__}))
    # missing the required "value" field -> ValidationError on load
    (entry / "model_dump.json").write_text(json.dumps({"name": "x"}))

    cache = StructuredOutputCache(use_cache=True)  # eager load must not raise

    assert not cache._memory_cache
    assert not entry.exists()


def test_disk_type_mismatch_evicts_entry() -> None:
    """A type-mismatched disk entry is evicted (rmtree) instead of crashing on unlink (#327)."""
    cache = StructuredOutputCache(use_cache=True)
    # plant a CacheModel at the key the cache derives for OtherModel inputs
    key = cache._get_cache_key(MESSAGES, OtherModel, PARAMS)
    cache._save_to_disk(key, CacheModel(name="x", value=1))
    cache._memory_cache.clear()

    assert cache._load_from_disk(key, OtherModel) is None
    assert not _get_structured_output_cache_path(key).exists()


@pytest.mark.asyncio
async def test_async_disk_type_mismatch_evicts_entry() -> None:
    """Async type-mismatched disk entry is evicted (rmtree) instead of crashing on unlink (#327)."""
    cache = StructuredOutputCache(use_cache=True)
    key = cache._get_cache_key(MESSAGES, OtherModel, PARAMS)
    await cache._save_to_disk_async(key, CacheModel(name="x", value=1))
    cache._memory_cache.clear()

    assert await cache._load_from_disk_async(key, OtherModel) is None
    assert not _get_structured_output_cache_path(key).exists()
