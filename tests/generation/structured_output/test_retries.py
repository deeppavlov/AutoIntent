"""Tests for structured output retry semantics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import httpx
import pytest
from pydantic import BaseModel, Field, model_validator

from autointent.generation import Generator, RetriesExceededError
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

    @model_validator(mode="after")
    def val_hobbies(self) -> Person:
        if len(self.hobbies) < 5:
            raise ValueError("it should has at least 5 hobbies")  # noqa: EM101, TRY003
        if self.status != "pending":
            raise ValueError("only pending status is supported now")  # noqa: EM101, TRY003
        if self.occupation != "office worker":
            raise ValueError("occupation should be `office worker`")  # noqa: EM101, TRY003
        return self


VALID_PERSON_JSON = json.dumps(
    {
        "reasoning": "ok",
        "name": "Alice Example",
        "age": 30,
        "email": "alice@example.com",
        "occupation": "office worker",
        "is_active": True,
        "status": "pending",
        "hobbies": ["reading", "hiking", "cooking", "gaming", "cycling"],
    }
)
INVALID_PERSON_JSON = json.dumps(
    {
        "reasoning": "bad",
        "name": "x",
        "age": 0,
        "email": "x@y",
        "occupation": "wrong",
        "is_active": True,
        "status": "active",
        "hobbies": [],
    }
)


def _resp(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest.fixture
def generator(respx_openai: MockRouter) -> Generator:
    return Generator(max_tokens=1000, use_cache=False)


class TestStructuredOutput:
    def test_structured_output_sync_success_with_enough_retries(
        self, generator: Generator, respx_openai: MockRouter
    ) -> None:
        respx_openai.post("/v1/chat/completions").mock(
            side_effect=[_resp(INVALID_PERSON_JSON), _resp(INVALID_PERSON_JSON), _resp(VALID_PERSON_JSON)]
        )
        result = generator.get_structured_output_sync(
            messages=[{"role": Role.USER, "content": "ok"}],
            output_model=Person,
            max_retries=5,
        )
        assert isinstance(result, Person)

    @pytest.mark.asyncio
    async def test_structured_output_async_success_with_enough_retries(
        self, generator: Generator, respx_openai: MockRouter
    ) -> None:
        respx_openai.post("/v1/chat/completions").mock(
            side_effect=[_resp(INVALID_PERSON_JSON), _resp(INVALID_PERSON_JSON), _resp(VALID_PERSON_JSON)]
        )
        result = await generator.get_structured_output_async(
            messages=[{"role": Role.USER, "content": "ok"}],
            output_model=Person,
            max_retries=5,
        )
        assert isinstance(result, Person)

    def test_structured_output_sync_failure_with_insufficient_retries(
        self, generator: Generator, respx_openai: MockRouter
    ) -> None:
        respx_openai.post("/v1/chat/completions").mock(return_value=_resp(INVALID_PERSON_JSON))
        with pytest.raises(RetriesExceededError):
            generator.get_structured_output_sync(
                messages=[{"role": Role.USER, "content": "ok"}],
                output_model=Person,
                max_retries=2,
            )

    @pytest.mark.asyncio
    async def test_structured_output_async_failure_with_insufficient_retries(
        self, generator: Generator, respx_openai: MockRouter
    ) -> None:
        respx_openai.post("/v1/chat/completions").mock(return_value=_resp(INVALID_PERSON_JSON))
        with pytest.raises(RetriesExceededError):
            await generator.get_structured_output_async(
                messages=[{"role": Role.USER, "content": "ok"}],
                output_model=Person,
                max_retries=2,
            )
