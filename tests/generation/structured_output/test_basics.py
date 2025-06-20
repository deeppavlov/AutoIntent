"""Tests for structured output functionality."""

import os
from typing import Literal

import pytest
from openai import APIConnectionError, BadRequestError
from pydantic import BaseModel, Field

from autointent.generation import Generator
from autointent.generation.chat_templates import Role


class Person(BaseModel):
    reasoning: str = Field(description="Some preliminary reasoning to plan fields' values")
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years", ge=0, le=150)
    email: str = Field(description="The person's email address")
    occupation: str = Field(description="The person's job or profession")
    is_active: bool = Field(description="Whether the person is currently active", default=True)
    status: Literal["active", "inactive", "pending"] = Field(description="Current status of the person")
    hobbies: list[str] = Field(description="List of the person's hobbies and interests")


@pytest.fixture
def generator():
    """Create a generator instance for testing."""
    return Generator(max_tokens=1000)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_MODEL_NAME"),
    reason="OPENAI_API_KEY and OPENAI_MODEL_NAME environment variables are required for this test",
)
@pytest.mark.parametrize("backend", ["openai", "vllm"])
class TestStructuredOutput:
    """Test structured output functionality for different backends."""

    def test_basic_chat_completion(self, generator, backend):
        """Test basic chat completion functionality."""
        try:
            response = generator.get_chat_completion(messages=[{"role": Role.USER, "content": "hi! tell me a joke"}])
            assert isinstance(response, str)
            assert len(response) > 0
        except (APIConnectionError, BadRequestError):
            pytest.skip(f"{backend} backend not available for testing")

    @pytest.mark.asyncio
    async def test_async_chat_completion(self, generator, backend):
        """Test async chat completion functionality."""
        try:
            response = await generator.get_chat_completion_async(
                messages=[{"role": Role.USER, "content": "hi! tell me a joke"}]
            )
            assert isinstance(response, str)
            assert len(response) > 0
        except (APIConnectionError, BadRequestError):
            pytest.skip(f"{backend} backend not available for testing")

    def test_structured_output(self, generator, backend):
        """Test that async structured output works without failing."""
        try:
            result = generator.get_structured_output_sync(
                messages=[{"role": Role.USER, "content": "Create a person"}],
                output_model=Person,
                backend=backend,
                max_retries=5,
            )

            assert isinstance(result, Person)
        except (APIConnectionError, BadRequestError):
            pytest.skip(f"{backend} backend not available for testing")

    @pytest.mark.asyncio
    async def test_structured_output_async(self, generator, backend):
        """Test that async structured output works without failing."""
        try:
            result = await generator.get_structured_output_async(
                messages=[{"role": Role.USER, "content": "Create a person"}],
                output_model=Person,
                backend=backend,
                max_retries=5,
            )

            assert isinstance(result, Person)
        except (APIConnectionError, BadRequestError):
            pytest.skip(f"{backend} backend not available for testing")
