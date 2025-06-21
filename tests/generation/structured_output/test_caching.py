"""Tests for structured output functionality."""

import os

import pytest
from pydantic import BaseModel, Field

from autointent.generation import Generator
from autointent.generation.chat_templates import Role


class SimpleModel(BaseModel):
    """Simple model for basic caching tests."""

    name: str = Field(description="A simple name")
    value: int = Field(description="A simple integer value")


@pytest.fixture
def generator_with_cache():
    """Create a generator instance for testing."""
    return Generator(max_tokens=1000, use_cache=True, temperature=2)  # increase randomness by increasing temperature


@pytest.fixture
def generator_without_cache():
    """Create a generator instance for testing."""
    return Generator(max_tokens=1000, use_cache=False, temperature=2)  # increase randomness by increasing temperature


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_MODEL_NAME"),
    reason="OPENAI_API_KEY and OPENAI_MODEL_NAME environment variables are required for this test",
)
@pytest.mark.asyncio
async def test_cache_hit(generator_with_cache, generator_without_cache):
    """Test that caching works correctly."""

    messages = [{"role": Role.USER, "content": "Create a random simple model"}]
    different_messages = [{"role": Role.USER, "content": "Create a person named John with value 333"}]

    # First call should miss cache and make API call
    result1 = await generator_with_cache.get_structured_output_async(
        messages=messages,
        output_model=SimpleModel,
        max_retries=3,
    )

    # Second identical call should hit cache
    result2 = await generator_with_cache.get_structured_output_async(
        messages=messages,
        output_model=SimpleModel,
        max_retries=3,
    )

    result3 = await generator_without_cache.get_structured_output_async(
        messages=different_messages,
        output_model=SimpleModel,
        max_retries=3,
    )

    # Results should be identical
    assert isinstance(result1, SimpleModel)
    assert isinstance(result2, SimpleModel)
    assert result1.name == result2.name
    assert result1.value == result2.value

    # cache is stored
    cached_res = generator_with_cache.cache.get(
        messages=messages,
        output_model=SimpleModel,
        backend="openai",
        generation_params=generator_with_cache.generation_params,
    )
    assert isinstance(cached_res, SimpleModel)
    assert cached_res.name == result1.name
    assert cached_res.value == result1.value

    # Third result should be different
    assert isinstance(result3, SimpleModel)
    assert result3.name != result1.name
    assert result3.value != result1.value
