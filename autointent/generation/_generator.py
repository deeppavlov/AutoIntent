"""Wrapper class for accessing OpenAI API."""

import os
from typing import Any, ClassVar, Literal, TypeVar

import openai
from dotenv import load_dotenv
from pydantic import BaseModel
from typing_extensions import assert_never

from autointent.generation.chat_templates import Message

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class Generator:
    """Wrapper class for accessing OpenAI API.

    Args:
        base_url: HTTP-endpoint for sending API requests to OpenAI API compatible server.
            Omit this to infer ``OPENAI_BASE_URL`` from environment.
        model_name: Name of LLM. Omit this to infer ``OPENAI_MODEL_NAME`` from environment.
        **generation_params: kwargs that will be sent with a request to the endpoint.
    """

    _default_generation_params: ClassVar[dict[str, Any]] = {
        "max_tokens": 150,
        "n": 1,
        "stop": None,
        "temperature": 0.7,
    }

    def __init__(self, base_url: str | None = None, model_name: str | None = None, **generation_params: Any) -> None:  # noqa: ANN401
        if not base_url:
            base_url = os.environ["OPENAI_BASE_URL"]
        if not model_name:
            model_name = os.environ["OPENAI_MODEL_NAME"]
        self.model_name = model_name
        self.client = openai.OpenAI(base_url=base_url)
        self.async_client = openai.AsyncOpenAI(base_url=base_url)
        self.generation_params = {
            **self._default_generation_params,
            **generation_params,
        }  #  https://stackoverflow.com/a/65539348

    def get_chat_completion(self, messages: list[Message]) -> str:
        """Prompt LLM and return its answer.

        Args:
            messages: List of messages to send to the model.
        """
        response = self.client.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]
            model=self.model_name,
            **self.generation_params,
        )
        return response.choices[0].message.content  # type: ignore[return-value]

    async def get_chat_completion_async(self, messages: list[Message]) -> str:
        """Prompt LLM and return its answer asynchronously.

        Args:
            messages: List of messages to send to the model.
        """
        response = await self.async_client.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]
            model=self.model_name,
            **self.generation_params,
        )
        return response.choices[0].message.content  # type: ignore[return-value]

    def get_structured_output(
        self, messages: list[Message], output_model: type[T], backend: Literal["openai", "vllm"] = "openai"
    ) -> T:
        """Prompt LLM and return structured output parsed into the provided Pydantic model.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            backend: Backend to use for structured output. Options: "openai" (uses beta.chat.completions.parse)
                    or "vllm" (uses guided_json with regular completions).

        Returns:
            Parsed response as an instance of the provided Pydantic model.
        """
        if backend == "openai":
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                response_format=output_model,
                **self.generation_params,
            )
            return response.choices[0].message.parsed
        if backend == "vllm":
            json_schema = output_model.model_json_schema()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                extra_body={"guided_json": json_schema},
                **self.generation_params,
            )
            content = response.choices[0].message.content
            return output_model.model_validate_json(content)

        assert_never(backend)

    async def get_structured_output_async(
        self, messages: list[Message], output_model: type[T], backend: Literal["openai", "vllm"] = "openai"
    ) -> T:
        """Prompt LLM and return structured output parsed into the provided Pydantic model asynchronously.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            backend: Backend to use for structured output. Options: "openai" (uses beta.chat.completions.parse)
                    or "vllm" (uses guided_json with regular completions).

        Returns:
            Parsed response as an instance of the provided Pydantic model.
        """
        if backend == "openai":
            response = await self.async_client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                response_format=output_model,
                **self.generation_params,
            )
            return response.choices[0].message.parsed
        if backend == "vllm":
            json_schema = output_model.model_json_schema()
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                extra_body={"guided_json": json_schema},
                **self.generation_params,
            )
            content = response.choices[0].message.content
            return output_model.model_validate_json(content)

        assert_never(backend)
