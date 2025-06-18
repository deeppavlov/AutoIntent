"""Wrapper class for accessing OpenAI API."""

import logging
import os
from textwrap import dedent
from typing import Any, ClassVar, Literal, TypeVar

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing_extensions import assert_never

from autointent.generation.chat_templates import Message

logger = logging.getLogger(__name__)

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

    def _create_retry_message(self, error_message: str, output_model: type[T]) -> Message:
        """Create a follow-up message for retry with error details and schema."""
        json_schema = output_model.model_json_schema()
        return {
            "role": "user",
            "content": dedent(
                f"""The previous response failed validation with the following error: {error_message}

                Please provide a valid JSON response that conforms to this schema:
                {json_schema}

                Make sure to:
                1. Follow the exact schema structure
                2. Use the correct data types for each field
                3. Include all required fields
                4. Ensure the response is valid JSON"""
            ),
        }

    async def _get_structured_output_openai_async(
        self, messages: list[Message], output_model: type[T]
    ) -> tuple[T | None, str | None]:
        res: T | None = None
        msg: str | None = None

        try:
            response = await self.async_client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                response_format=output_model,
                **self.generation_params,
            )
            res = response.choices[0].message.parsed
        except (ValidationError, ValueError) as e:
            msg = f"Failed to obtain structured output for model {self.model_name} and messages {messages}: {e!s}"
            logger.warning(msg)
        else:
            if res is None:
                msg = "For some reason output wasn't parsed."
                logger.warning(msg)

        return res, msg

    async def _get_structured_output_vllm_async(
        self, messages: list[Message], output_model: type[T]
    ) -> tuple[T | None, str | None]:
        """https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html."""
        res: T | None = None
        msg: str | None = None

        try:
            json_schema = output_model.model_json_schema()
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                extra_body={"guided_json": json_schema},
                **self.generation_params,
            )
            content: str = response.choices[0].message.content  # type: ignore[assignment]
            res = output_model.model_validate_json(content)
        except (ValidationError, ValueError) as e:
            msg = f"Failed to obtain structured output for model {self.model_name} and messages {messages}: {e!s}"
            logger.warning(msg)

        return res, msg

    async def get_structured_output_async(
        self,
        messages: list[Message],
        output_model: type[T],
        backend: Literal["openai", "vllm"] = "openai",
        max_retries: int = 3,
    ) -> T:
        """Prompt LLM and return structured output parsed into the provided Pydantic model asynchronously.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            backend: Backend to use for structured output. Options: "openai" (uses beta.chat.completions.parse)
                    or "vllm" (uses guided_json with regular completions).
            max_retries: Maximum number of retry attempts for failed validations.

        Returns:
            Parsed response as an instance of the provided Pydantic model.
        """
        current_messages = messages.copy()
        res: T | None = None

        for _ in range(max_retries + 1):
            if backend == "openai":
                res, error = await self._get_structured_output_openai_async(current_messages, output_model)
            elif backend == "vllm":
                res, error = await self._get_structured_output_vllm_async(current_messages, output_model)
            else:
                assert_never(backend)

            if res is not None:
                break

            if error is None:
                msg = "Structured output returned None but no error was caught."
                logger.exception(msg)
                raise RuntimeError(msg)

            current_messages.append(self._create_retry_message(error, output_model))

        if res is None:
            msg = (
                f"Failed to generate valid structured output after {max_retries + 1} attempts.\n"
                f"Messages: {current_messages}"
            )
            logger.exception(msg)
            raise RuntimeError(msg)

        return res

    def _get_structured_output_openai_sync(
        self, messages: list[Message], output_model: type[T]
    ) -> tuple[T | None, str | None]:
        res: T | None = None
        msg: str | None = None

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                response_format=output_model,
                **self.generation_params,
            )
            res = response.choices[0].message.parsed
        except (ValidationError, ValueError) as e:
            msg = f"Failed to obtain structured output for model {self.model_name} and messages {messages}: {e!s}"
            logger.warning(msg)
        else:
            if res is None:
                msg = "For some reason output wasn't parsed."
                logger.warning(msg)

        return res, msg

    def _get_structured_output_vllm_sync(
        self, messages: list[Message], output_model: type[T]
    ) -> tuple[T | None, str | None]:
        """https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html."""
        res: T | None = None
        msg: str | None = None

        try:
            json_schema = output_model.model_json_schema()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                extra_body={"guided_json": json_schema},
                **self.generation_params,
            )
            content: str = response.choices[0].message.content  # type: ignore[assignment]
            res = output_model.model_validate_json(content)
        except (ValidationError, ValueError) as e:
            msg = f"Failed to obtain structured output for model {self.model_name} and messages {messages}: {e!s}"
            logger.warning(msg)

        return res, msg

    def get_structured_output_sync(
        self,
        messages: list[Message],
        output_model: type[T],
        backend: Literal["openai", "vllm"] = "openai",
        max_retries: int = 3,
    ) -> T:
        """Prompt LLM and return structured output parsed into the provided Pydantic model.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            backend: Backend to use for structured output. Options: "openai" (uses beta.chat.completions.parse)
                    or "vllm" (uses guided_json with regular completions).
            max_retries: Maximum number of retry attempts for failed validations.

        Returns:
            Parsed response as an instance of the provided Pydantic model.
        """
        current_messages = messages.copy()
        res: T | None = None

        for _ in range(max_retries + 1):
            if backend == "openai":
                res, error = self._get_structured_output_openai_sync(current_messages, output_model)
            elif backend == "vllm":
                res, error = self._get_structured_output_vllm_sync(current_messages, output_model)
            else:
                assert_never(backend)

            if res is not None:
                break

            if error is None:
                msg = "Structured output returned None but no error was caught."
                logger.exception(msg)
                raise RuntimeError(msg)

            current_messages.append(self._create_retry_message(error, output_model))

        if res is None:
            msg = (
                f"Failed to generate valid structured output after {max_retries + 1} attempts.\n"
                f"Messages: {current_messages}"
            )
            logger.exception(msg)
            raise RuntimeError(msg)

        return res
