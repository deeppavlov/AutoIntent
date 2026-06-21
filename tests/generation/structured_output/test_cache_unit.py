"""Unit tests for StructuredOutputCache memory/disk/async behavior (no LLM)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from autointent.generation._cache import StructuredOutputCache
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
