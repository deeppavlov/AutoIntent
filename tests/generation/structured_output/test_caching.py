"""Tests for structured output functionality."""

import os
from pathlib import Path

import pytest
from openai import APIConnectionError, BadRequestError
from pydantic import BaseModel, Field

from autointent.generation import Generator
from autointent.generation.chat_templates import Role


class SimpleModel(BaseModel):
    """Simple model for basic caching tests."""

    name: str = Field(description="A simple name")
    value: int = Field(description="A simple integer value")


class SimpleModelNoExtra(BaseModel, extra="forbid"):
    """Simple model for basic caching tests."""

    name: str = Field(description="A simple name")
    value: int = Field(description="A simple integer value")


@pytest.fixture
def generator_with_cache():
    """Create a generator instance for testing."""
    return Generator(max_tokens=1000, use_cache=True, temperature=10)  # increase randomness by increasing temperature


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_MODEL_NAME"),
    reason="OPENAI_API_KEY and OPENAI_MODEL_NAME environment variables are required for this test",
)
class TestStructuredOutputCaching:
    """Test caching functionality for structured outputs using async methods."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, generator_with_cache):
        """Test that caching works correctly."""

        messages = [{"role": Role.USER, "content": "Create a random simple model"}]
        different_messages = [{"role": Role.USER, "content": "Create a person named John with value 333"}]
        try:
            # First call should miss cache and make API call
            result1 = await generator_with_cache.get_structured_output_async(
                messages=messages,
                output_model=SimpleModel,
                backend="openai",
                max_retries=3,
            )

            # Second identical call should hit cache
            result2 = await generator_with_cache.get_structured_output_async(
                messages=messages,
                output_model=SimpleModel,
                backend="openai",
                max_retries=3,
            )

            result3 = await generator_with_cache.get_structured_output_async(
                messages=different_messages,
                output_model=SimpleModel,
                backend="openai",
                max_retries=3,
            )

            # Results should be identical
            assert isinstance(result1, SimpleModel)
            assert isinstance(result2, SimpleModel)
            assert result1.name == result2.name
            assert result1.value == result2.value

            # Second result is taken from cache
            assert hasattr(result2, "__cache_source")
            assert Path(result2.__cache_source).exists()

            # Third result should be different
            assert isinstance(result3, SimpleModel)
            assert result3.name != result1.name
            assert result3.value != result1.value

        except (APIConnectionError, BadRequestError):
            pytest.skip("OpenAI backend not available for testing")

    @pytest.mark.asyncio
    async def test_cache_source_not_added_to_models_without_extra_fields(self, generator_with_cache):
        """Test that cache source field is not added to models that don't allow extra fields."""

        messages = [{"role": Role.USER, "content": "Create a random simple model"}]

        try:
            # First call should miss cache
            result1 = await generator_with_cache.get_structured_output_async(
                messages=messages,
                output_model=SimpleModelNoExtra,
                backend="openai",
                max_retries=3,
            )

            # Second call should hit cache but not have cache source field
            result2 = await generator_with_cache.get_structured_output_async(
                messages=messages,
                output_model=SimpleModelNoExtra,
                backend="openai",
                max_retries=3,
            )

            assert isinstance(result1, SimpleModelNoExtra)
            assert isinstance(result2, SimpleModelNoExtra)

            # Neither result should have cache source field
            assert not hasattr(result1, "__cache_source")
            assert not hasattr(result2, "__cache_source")

            # Results shouldn't be identical
            assert result1.name == result2.name
            assert result1.value == result2.value

        except (APIConnectionError, BadRequestError):
            pytest.skip("OpenAI backend not available for testing")
