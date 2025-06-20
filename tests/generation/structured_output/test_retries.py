"""Tests for structured output functionality."""

import os
from typing import Literal

import pytest
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
    return Generator(max_tokens=1000, use_cache=False)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_MODEL_NAME"),
    reason="OPENAI_API_KEY and OPENAI_MODEL_NAME environment variables are required for this test",
)
class TestStructuredOutput:
    """Test structured output functionality for different backends."""

    def test_structured_output_sync_success_with_enough_retries(self, generator):
        """Test structured output sync that succeeds with enough retries."""
        result = generator.get_structured_output_sync(
            messages=[{"role": Role.USER, "content": "How would a nice student look like?"}],
            output_model=Person,
            max_retries=5,
        )

        assert isinstance(result, Person)

    @pytest.mark.asyncio
    async def test_structured_output_async_success_with_enough_retries(self, generator):
        """Test structured output async that succeeds with enough retries."""
        result = await generator.get_structured_output_async(
            messages=[{"role": Role.USER, "content": "How would a nice student look like?"}],
            output_model=Person,
            max_retries=5,
        )

        assert isinstance(result, Person)

    def test_structured_output_sync_failure_with_insufficient_retries(self, generator):
        """Test structured output sync that fails with insufficient retries."""
        with pytest.raises(RuntimeError, match="Failed to generate valid structured output after 3 attempts"):
            generator.get_structured_output_sync(
                messages=[{"role": Role.USER, "content": "How would a nice student look like?"}],
                output_model=Person,
                max_retries=2,
            )

    @pytest.mark.asyncio
    async def test_structured_output_async_failure_with_insufficient_retries(self, generator):
        """Test structured output async that fails with insufficient retries."""
        with pytest.raises(RuntimeError, match="Failed to generate valid structured output after 3 attempts"):
            await generator.get_structured_output_async(
                messages=[{"role": Role.USER, "content": "How would a nice student look like?"}],
                output_model=Person,
                max_retries=2,
            )
