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


MODEL_NAME = "test-model"
BASE_URL: str | None = None


def test_set_get_memory_roundtrip() -> None:
    cache = StructuredOutputCache(use_cache=True)
    result = CacheModel(name="a", value=1)
    cache.set(MESSAGES, CacheModel, PARAMS, result, MODEL_NAME, BASE_URL)
    assert cache.get(MESSAGES, CacheModel, PARAMS, MODEL_NAME, BASE_URL) == result


def test_disabled_cache_is_noop() -> None:
    cache = StructuredOutputCache(use_cache=False)
    cache.set(MESSAGES, CacheModel, PARAMS, CacheModel(name="a", value=1), MODEL_NAME, BASE_URL)
    assert cache.get(MESSAGES, CacheModel, PARAMS, MODEL_NAME, BASE_URL) is None


def test_get_misses_for_unknown_key() -> None:
    cache = StructuredOutputCache(use_cache=True)
    assert cache.get(MESSAGES, CacheModel, PARAMS, MODEL_NAME, BASE_URL) is None


def test_get_loads_from_disk_in_fresh_instance() -> None:
    """A second instance has empty memory and must read the entry back from disk."""
    StructuredOutputCache(use_cache=True).set(
        MESSAGES, CacheModel, PARAMS, CacheModel(name="x", value=9), MODEL_NAME, BASE_URL
    )

    fresh = StructuredOutputCache(use_cache=True)
    fresh._memory_cache.clear()  # force the disk path even if eager load changes
    loaded = fresh.get(MESSAGES, CacheModel, PARAMS, MODEL_NAME, BASE_URL)
    assert isinstance(loaded, CacheModel)
    assert loaded.value == 9
    # disk hit populates the memory cache for next time
    assert fresh._memory_cache


def test_memory_type_mismatch_evicts() -> None:
    cache = StructuredOutputCache(use_cache=True)
    key = cache._get_cache_key(MESSAGES, CacheModel, PARAMS, MODEL_NAME, BASE_URL)
    cache._memory_cache[key] = OtherModel(text="wrong")
    assert cache._check_memory_cache(key, CacheModel) is None
    assert key not in cache._memory_cache


@pytest.mark.asyncio
async def test_async_set_get_roundtrip() -> None:
    cache = StructuredOutputCache(use_cache=True)
    result = CacheModel(name="async", value=7)
    await cache.set_async(MESSAGES, CacheModel, PARAMS, result, MODEL_NAME, BASE_URL)
    assert await cache.get_async(MESSAGES, CacheModel, PARAMS, MODEL_NAME, BASE_URL) == result


@pytest.mark.asyncio
async def test_async_get_loads_from_disk() -> None:
    await StructuredOutputCache(use_cache=True).set_async(
        MESSAGES, CacheModel, PARAMS, CacheModel(name="d", value=3), MODEL_NAME, BASE_URL
    )

    fresh = StructuredOutputCache(use_cache=True)
    fresh._memory_cache.clear()
    loaded = await fresh.get_async(MESSAGES, CacheModel, PARAMS, MODEL_NAME, BASE_URL)
    assert isinstance(loaded, CacheModel)
    assert loaded.value == 3


@pytest.mark.asyncio
async def test_async_disabled_cache_is_noop() -> None:
    cache = StructuredOutputCache(use_cache=False)
    await cache.set_async(MESSAGES, CacheModel, PARAMS, CacheModel(name="a", value=1), MODEL_NAME, BASE_URL)
    assert await cache.get_async(MESSAGES, CacheModel, PARAMS, MODEL_NAME, BASE_URL) is None


# --- Regression tests for the on-disk-cache bugs (#326 eager load, #327 eviction) ---
# Disk entries are directories (PydanticModelDumper writes class_info.json +
# model_dump.json), so eager load must collect directories and eviction must
# rmtree rather than unlink.


def test_eager_load_populates_memory_from_disk() -> None:
    """A fresh instance eagerly batch-loads existing on-disk entries into memory (#326)."""
    StructuredOutputCache(use_cache=True).set(
        MESSAGES, CacheModel, PARAMS, CacheModel(name="x", value=9), MODEL_NAME, BASE_URL
    )

    fresh = StructuredOutputCache(use_cache=True)
    key = fresh._get_cache_key(MESSAGES, CacheModel, PARAMS, MODEL_NAME, BASE_URL)

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
    key = cache._get_cache_key(MESSAGES, OtherModel, PARAMS, MODEL_NAME, BASE_URL)
    cache._save_to_disk(key, CacheModel(name="x", value=1))
    cache._memory_cache.clear()

    assert cache._load_from_disk(key, OtherModel) is None
    assert not _get_structured_output_cache_path(key).exists()


@pytest.mark.asyncio
async def test_async_disk_type_mismatch_evicts_entry() -> None:
    """Async type-mismatched disk entry is evicted (rmtree) instead of crashing on unlink (#327)."""
    cache = StructuredOutputCache(use_cache=True)
    key = cache._get_cache_key(MESSAGES, OtherModel, PARAMS, MODEL_NAME, BASE_URL)
    await cache._save_to_disk_async(key, CacheModel(name="x", value=1))
    cache._memory_cache.clear()

    assert await cache._load_from_disk_async(key, OtherModel) is None
    assert not _get_structured_output_cache_path(key).exists()


# --- Regression tests for model-identity cache collision (#334) ---


def test_different_model_names_do_not_collide() -> None:
    """Two generators with different model_name must NOT share a cache entry (#334)."""
    result_a = CacheModel(name="from-model-a", value=1)
    result_b = CacheModel(name="from-model-b", value=2)

    cache = StructuredOutputCache(use_cache=True)
    cache.set(MESSAGES, CacheModel, PARAMS, result_a, model_name="model-a", base_url=None)
    cache.set(MESSAGES, CacheModel, PARAMS, result_b, model_name="model-b", base_url=None)

    hit_a = cache.get(MESSAGES, CacheModel, PARAMS, model_name="model-a", base_url=None)
    hit_b = cache.get(MESSAGES, CacheModel, PARAMS, model_name="model-b", base_url=None)

    assert hit_a == result_a, "model-a should get its own cached value"
    assert hit_b == result_b, "model-b must NOT get model-a's value"


def test_different_base_urls_do_not_collide() -> None:
    """Two generators with different base_url must NOT share a cache entry (#334)."""
    result_x = CacheModel(name="from-url-x", value=10)
    result_y = CacheModel(name="from-url-y", value=20)

    cache = StructuredOutputCache(use_cache=True)
    cache.set(MESSAGES, CacheModel, PARAMS, result_x, model_name="gpt-4o", base_url="http://host-x/v1")
    cache.set(MESSAGES, CacheModel, PARAMS, result_y, model_name="gpt-4o", base_url="http://host-y/v1")

    hit_x = cache.get(MESSAGES, CacheModel, PARAMS, model_name="gpt-4o", base_url="http://host-x/v1")
    hit_y = cache.get(MESSAGES, CacheModel, PARAMS, model_name="gpt-4o", base_url="http://host-y/v1")

    assert hit_x == result_x, "host-x should get its own cached value"
    assert hit_y == result_y, "host-y must NOT get host-x's value"


def test_same_identity_still_hits_cache() -> None:
    """Same model_name + base_url + inputs must continue to yield a cache hit (#334)."""
    result = CacheModel(name="same", value=42)

    cache = StructuredOutputCache(use_cache=True)
    cache.set(MESSAGES, CacheModel, PARAMS, result, model_name="gpt-4o", base_url="http://host/v1")

    hit = cache.get(MESSAGES, CacheModel, PARAMS, model_name="gpt-4o", base_url="http://host/v1")
    assert hit == result


@pytest.mark.asyncio
async def test_async_different_model_names_do_not_collide() -> None:
    """Async paths: two model names must NOT collide (#334)."""
    result_a = CacheModel(name="async-a", value=1)
    result_b = CacheModel(name="async-b", value=2)

    cache = StructuredOutputCache(use_cache=True)
    await cache.set_async(MESSAGES, CacheModel, PARAMS, result_a, model_name="async-model-a", base_url=None)
    await cache.set_async(MESSAGES, CacheModel, PARAMS, result_b, model_name="async-model-b", base_url=None)

    hit_a = await cache.get_async(MESSAGES, CacheModel, PARAMS, model_name="async-model-a", base_url=None)
    hit_b = await cache.get_async(MESSAGES, CacheModel, PARAMS, model_name="async-model-b", base_url=None)

    assert hit_a == result_a
    assert hit_b == result_b


@pytest.mark.asyncio
async def test_async_same_identity_still_hits_cache() -> None:
    """Async paths: same identity must still yield a hit (#334)."""
    result = CacheModel(name="async-same", value=99)

    cache = StructuredOutputCache(use_cache=True)
    await cache.set_async(MESSAGES, CacheModel, PARAMS, result, model_name="gpt-4o", base_url=None)

    hit = await cache.get_async(MESSAGES, CacheModel, PARAMS, model_name="gpt-4o", base_url=None)
    assert hit == result
