"""Wrapper class for accessing OpenAI API."""

import os
from typing import Any, ClassVar

import openai
from dotenv import load_dotenv

from autointent.generation.chat_templates import Message

load_dotenv()


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
