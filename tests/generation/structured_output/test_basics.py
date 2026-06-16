"""Tests for structured output functionality."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import httpx
import pytest
from pydantic import BaseModel, Field

from autointent.generation import Generator
from autointent.generation.chat_templates import Role

if TYPE_CHECKING:
    from respx.router import MockRouter


class Person(BaseModel):
    reasoning: str = Field(description="Some preliminary reasoning to plan fields' values")
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years", ge=0, le=150)
    email: str = Field(description="The person's email address")
    occupation: str = Field(description="The person's job or profession")
    is_active: bool = Field(description="Whether the person is currently active", default=True)
    status: Literal["active", "inactive", "pending"] = Field(description="Current status of the person")
    hobbies: list[str] = Field(description="List of the person's hobbies and interests")


VALID_PERSON_JSON = json.dumps(
    {
        "reasoning": "ok",
        "name": "Alice Example",
        "age": 30,
        "email": "alice@example.com",
        "occupation": "engineer",
        "is_active": True,
        "status": "active",
        "hobbies": ["reading"],
    }
)


def _chat_completion_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest.fixture
def generator(respx_openai: MockRouter) -> Generator:
    """Create a generator instance for testing."""
    return Generator(max_tokens=1000, use_cache=False)


class TestStructuredOutput:
    """Test structured output functionality for different backends."""

    def test_basic_chat_completion(self, generator: Generator, respx_openai: MockRouter) -> None:
        respx_openai.post("/v1/chat/completions").mock(return_value=_chat_completion_response("hi! here's a joke"))
        response = generator.get_chat_completion(messages=[{"role": Role.USER, "content": "hi! tell me a joke"}])
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_async_chat_completion(self, generator: Generator, respx_openai: MockRouter) -> None:
        respx_openai.post("/v1/chat/completions").mock(return_value=_chat_completion_response("hi! here's a joke"))
        response = await generator.get_chat_completion_async(
            messages=[{"role": Role.USER, "content": "hi! tell me a joke"}]
        )
        assert isinstance(response, str)
        assert len(response) > 0

    def test_structured_output(self, generator: Generator, respx_openai: MockRouter) -> None:
        respx_openai.post("/v1/chat/completions").mock(return_value=_chat_completion_response(VALID_PERSON_JSON))
        result = generator.get_structured_output_sync(
            messages=[{"role": Role.USER, "content": "Create a person"}],
            output_model=Person,
            max_retries=5,
        )
        assert isinstance(result, Person)

    @pytest.mark.asyncio
    async def test_structured_output_async(self, generator: Generator, respx_openai: MockRouter) -> None:
        respx_openai.post("/v1/chat/completions").mock(return_value=_chat_completion_response(VALID_PERSON_JSON))
        result = await generator.get_structured_output_async(
            messages=[{"role": Role.USER, "content": "Create a person"}],
            output_model=Person,
            max_retries=5,
        )
        assert isinstance(result, Person)
