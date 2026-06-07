"""Tests for Generator cache semantics."""

import json

import httpx
import pytest
from pydantic import BaseModel, Field

from autointent.generation import Generator
from autointent.generation.chat_templates import Role


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Redirect the structured-output disk cache to a fresh tmp dir each test."""
    monkeypatch.setattr("autointent.generation._cache.user_cache_dir", lambda *_: str(tmp_path))


class SimpleModel(BaseModel):
    name: str = Field(description="A simple name")
    value: int = Field(description="A simple integer value")


def _resp(name: str, value: int) -> httpx.Response:
    payload = json.dumps({"name": name, "value": value})
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": payload}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest.fixture
def generator_with_cache(respx_openai):
    return Generator(max_tokens=1000, use_cache=True, temperature=2)


@pytest.fixture
def generator_without_cache(respx_openai):
    return Generator(max_tokens=1000, use_cache=False, temperature=2)


@pytest.mark.asyncio
async def test_cache_hit(generator_with_cache, generator_without_cache, respx_openai):
    messages = [{"role": Role.USER, "content": "Create a random simple model"}]
    different_messages = [{"role": Role.USER, "content": "Create a person named John with value 333"}]

    route = respx_openai.post("/v1/chat/completions").mock(
        side_effect=[
            _resp("Alpha", 1),
            _resp("Beta", 2),
        ]
    )

    result1 = await generator_with_cache.get_structured_output_async(
        messages=messages, output_model=SimpleModel, max_retries=3
    )
    result2 = await generator_with_cache.get_structured_output_async(
        messages=messages, output_model=SimpleModel, max_retries=3
    )
    result3 = await generator_without_cache.get_structured_output_async(
        messages=different_messages, output_model=SimpleModel, max_retries=3
    )

    assert isinstance(result1, SimpleModel)
    assert isinstance(result2, SimpleModel)
    assert isinstance(result3, SimpleModel)
    assert result1.name == result2.name == "Alpha"
    assert result1.value == result2.value == 1
    assert result3.name == "Beta"
    assert result3.value == 2

    assert route.call_count == 2

    cached_res = generator_with_cache.cache.get(
        messages=messages,
        output_model=SimpleModel,
        generation_params=generator_with_cache.generation_params,
    )
    assert isinstance(cached_res, SimpleModel)
    assert cached_res.name == result1.name
    assert cached_res.value == result1.value
