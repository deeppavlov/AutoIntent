"""Tests for structured output functionality."""

import os
from typing import Literal

import pytest
from openai import APIConnectionError, BadRequestError
from pydantic import BaseModel, Field, model_validator

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

    @model_validator(mode="after")
    def val_hobbies(self) -> "Person":
        if len(self.hobbies) < 5:
            raise ValueError("it should has at least 5 hobbies")  # noqa: EM101, TRY003
        if self.status != "pending":
            raise ValueError("only pending status is supported now")  # noqa: EM101, TRY003
        if self.occupation != "office worker":
            raise ValueError("occupation should be `office worker`")  # noqa: EM101, TRY003
        return self


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

    def test_structured_output_sync_success_with_enough_retries(self, generator, backend):
        """Test structured output sync that succeeds with enough retries."""
        try:
            result = generator.get_structured_output_sync(
                messages=[{"role": Role.USER, "content": "How would a nice student look like?"}],
                output_model=Person,
                backend=backend,
                max_retries=5,
            )

            assert isinstance(result, Person)
        except (APIConnectionError, BadRequestError):
            pytest.skip(f"{backend} backend not available for testing")

    @pytest.mark.asyncio
    async def test_structured_output_async_success_with_enough_retries(self, generator, backend):
        """Test structured output async that succeeds with enough retries."""
        try:
            result = await generator.get_structured_output_async(
                messages=[{"role": Role.USER, "content": "How would a nice student look like?"}],
                output_model=Person,
                backend=backend,
                max_retries=5,
            )

            assert isinstance(result, Person)
        except (APIConnectionError, BadRequestError):
            pytest.skip(f"{backend} backend not available for testing")

    def test_structured_output_sync_failure_with_insufficient_retries(self, generator, backend):
        """Test structured output sync that fails with insufficient retries."""
        try:
            with pytest.raises(RuntimeError, match="Failed to generate valid structured output after 3 attempts"):
                generator.get_structured_output_sync(
                    messages=[{"role": Role.USER, "content": "How would a nice student look like?"}],
                    output_model=Person,
                    backend=backend,
                    max_retries=2,
                )
        except (APIConnectionError, BadRequestError):
            pytest.skip(f"{backend} backend not available for testing")

    @pytest.mark.asyncio
    async def test_structured_output_async_failure_with_insufficient_retries(self, generator, backend):
        """Test structured output async that fails with insufficient retries."""
        try:
            with pytest.raises(RuntimeError, match="Failed to generate valid structured output after 3 attempts"):
                await generator.get_structured_output_async(
                    messages=[{"role": Role.USER, "content": "How would a nice student look like?"}],
                    output_model=Person,
                    backend=backend,
                    max_retries=2,
                )
        except (APIConnectionError, BadRequestError):
            pytest.skip(f"{backend} backend not available for testing")

    def test_structured_output_validation_requirements(self, generator, backend):
        """Test that structured output respects validation requirements."""
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
    async def test_structured_output_async_validation_requirements(self, generator, backend):
        """Test that async structured output respects validation requirements."""
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

    def test_structured_output_with_complex_prompt(self, generator, backend):
        """Test structured output with a complex prompt."""
        try:
            complex_prompt = """
            Create a person who is a student but must meet specific requirements:
            - They should be interested in technology
            - They should have diverse hobbies
            - They should be active in their community
            """

            result = generator.get_structured_output_sync(
                messages=[{"role": Role.USER, "content": complex_prompt}],
                output_model=Person,
                backend=backend,
                max_retries=5,
            )

            assert isinstance(result, Person)
            assert result.reasoning is not None
            assert len(result.reasoning) > 0
        except (APIConnectionError, BadRequestError):
            pytest.skip(f"{backend} backend not available for testing")

    def test_structured_output_with_minimal_prompt(self, generator, backend):
        """Test structured output with a minimal prompt."""
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

    def test_structured_output_retry_mechanism(self, generator, backend):
        """Test that retry mechanism works correctly."""
        try:
            # This test verifies that the retry mechanism is working
            # by using a prompt that might initially fail validation
            result = generator.get_structured_output_sync(
                messages=[{"role": Role.USER, "content": "Describe a student"}],
                output_model=Person,
                backend=backend,
                max_retries=3,
            )

            assert isinstance(result, Person)
        except (APIConnectionError, BadRequestError):
            pytest.skip(f"{backend} backend not available for testing")
