"""Wrapper class for accessing OpenAI API."""

import json
import logging
import os
import pickle
from pathlib import Path
from textwrap import dedent
from typing import Any, ClassVar, Literal, TypeVar

import openai
from appdirs import user_cache_dir
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing_extensions import assert_never

from autointent._hash import Hasher
from autointent.generation.chat_templates import Message, Role

logger = logging.getLogger(__name__)

load_dotenv()

T = TypeVar("T", bound=BaseModel)
"""Type variable for Pydantic models used in structured output generation."""


def _get_structured_output_cache_path(filename: str) -> Path:
    """Get the path to the structured output cache file.

    This function constructs the full path to a cache file stored
    in a specific directory under the user's home directory. The cache
    file is named based on the provided filename, with the `.pkl` extension
    added.

    Args:
        filename: The name of the cache file (without extension).

    Returns:
        The full path to the cache file.
    """
    return Path(user_cache_dir("autointent")) / "structured_outputs" / f"{filename}.pkl"


class StructuredOutputCache:
    """Cache for structured output results."""

    def __init__(self, use_cache: bool = True) -> None:
        """Initialize the cache.

        Args:
            use_cache: Whether to use caching.
        """
        self.use_cache = use_cache

    def _get_cache_key(
        self, messages: list[Message], output_model: type[T], backend: str, generation_params: dict[str, Any]
    ) -> str:
        """Generate a cache key for the given parameters.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            backend: Backend to use for structured output.
            generation_params: Generation parameters.

        Returns:
            Cache key as a hexadecimal string.
        """
        hasher = Hasher()
        hasher.update(json.dumps(messages))
        hasher.update(json.dumps(output_model.model_json_schema()))
        hasher.update(backend)
        hasher.update(json.dumps(generation_params))
        return hasher.hexdigest()

    def get(
        self, messages: list[Message], output_model: type[T], backend: str, generation_params: dict[str, Any]
    ) -> T | None:
        """Get cached result if available.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            backend: Backend to use for structured output.
            generation_params: Generation parameters.

        Returns:
            Cached result if available, None otherwise.
        """
        if not self.use_cache:
            return None

        cache_key = self._get_cache_key(messages, output_model, backend, generation_params)
        cache_path = _get_structured_output_cache_path(cache_key)

        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    cached_data = pickle.load(f)
                    # Validate that the cached data matches the expected model
                    if isinstance(cached_data, output_model):
                        logger.debug("Using cached structured output for key: %s", cache_key)
                        return cached_data

                    logger.warning("Cached data type mismatch, removing invalid cache")
                    cache_path.unlink()
            except (pickle.PickleError, EOFError) as e:
                logger.warning("Failed to load cached structured output: %s", e)
                cache_path.unlink(missing_ok=True)

        return None

    def set(
        self, messages: list[Message], output_model: type[T], backend: str, generation_params: dict[str, Any], result: T
    ) -> None:
        """Cache the result.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.
            backend: Backend to use for structured output.
            generation_params: Generation parameters.
            result: The result to cache.
        """
        if not self.use_cache:
            return

        cache_key = self._get_cache_key(messages, output_model, backend, generation_params)
        cache_path = _get_structured_output_cache_path(cache_key)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(result, f)
        logger.debug("Cached structured output for key: %s", cache_key)



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
    """Default generation parameters for API requests."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        use_cache: bool = True,
        **generation_params: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the Generator with API configuration.

        Args:
            base_url: OpenAI API compatible server URL.
            model_name: Name of the language model to use.
            use_cache: Whether to use caching for structured outputs.
            **generation_params: Additional generation parameters to override defaults passed to OpenAI completions API.
        """
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        model_name = model_name or os.getenv("OPENAI_MODEL_NAME")
        if model_name is None:
            msg = "Specify model_name arg or OPENAI_MODEL_NAME environment variable"
            raise ValueError(msg)
        self.model_name = model_name
        self.client = openai.OpenAI(base_url=base_url)
        self.async_client = openai.AsyncOpenAI(base_url=base_url)
        self.generation_params = {
            **self._default_generation_params,
            **generation_params,
        }  #  https://stackoverflow.com/a/65539348
        self.cache = StructuredOutputCache(use_cache=use_cache)

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

    def _create_retry_messages(self, error_message: str, raw: str | None) -> list[Message]:
        """Create a follow-up message for retry with error details and schema."""
        res: list[Message] = []
        if raw is not None:
            res.append({"role": Role.ASSISTANT, "content": raw})
        res.append(
            {
                "role": "user",
                "content": dedent(
                    f"""The previous response failed validation with the following error: {error_message}

                Make sure to:
                1. Follow the exact schema structure
                2. Use the correct data types for each field
                3. Include all required fields
                4. Ensure the response is valid JSON"""
                ),
            }
        )
        return res

    async def _get_structured_output_openai_async(
        self, messages: list[Message], output_model: type[T]
    ) -> tuple[T | None, str | None, str | None]:
        """Get structured output using OpenAI's beta parse endpoint asynchronously.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.

        Returns:
            Tuple of (parsed_result, error_message, raw_response).
        """
        res: T | None = None
        msg: str | None = None
        raw: str | None = None

        try:
            response = await self.async_client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                response_format=output_model,
                **self.generation_params,
            )
            raw = response.choices[0].message.content
            res = response.choices[0].message.parsed
        except (ValidationError, ValueError) as e:
            msg = f"Failed to obtain structured output for model {self.model_name} and messages {messages}: {e!s}"
            logger.warning(msg)
        else:
            if res is None:
                msg = "For some reason output wasn't parsed."
                logger.warning(msg)

        return res, msg, raw

    async def _get_structured_output_vllm_async(
        self, messages: list[Message], output_model: type[T]
    ) -> tuple[T | None, str | None, str | None]:
        """https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html."""
        res: T | None = None
        msg: str | None = None
        raw: str | None = None

        try:
            json_schema = output_model.model_json_schema()
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                extra_body={"guided_json": json_schema},
                **self.generation_params,
            )
            raw = response.choices[0].message.content
            res = output_model.model_validate_json(raw)  # type: ignore[arg-type]
        except (ValidationError, ValueError) as e:
            msg = f"Failed to obtain structured output for model {self.model_name} and messages {messages}: {e!s}"
            logger.warning(msg)

        return res, msg, raw

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
        # Check cache first
        cached_result = self.cache.get(messages, output_model, backend, self.generation_params)
        if cached_result is not None:
            return cached_result

        current_messages = messages.copy()
        res: T | None = None

        for _ in range(max_retries + 1):
            if backend == "openai":
                res, error, raw = await self._get_structured_output_openai_async(current_messages, output_model)
            elif backend == "vllm":
                res, error, raw = await self._get_structured_output_vllm_async(current_messages, output_model)
            else:
                assert_never(backend)

            if res is not None:
                break

            if error is None:
                msg = "Structured output returned None but no error was caught."
                logger.exception(msg)
                raise RuntimeError(msg)

            current_messages.extend(self._create_retry_messages(error, raw))

        if res is None:
            msg = (
                f"Failed to generate valid structured output after {max_retries + 1} attempts.\n"
                f"Messages: {current_messages}"
            )
            logger.exception(msg)
            raise RuntimeError(msg)

        # Cache the successful result
        self.cache.set(messages, output_model, backend, self.generation_params, res)

        return res

    def _get_structured_output_openai_sync(
        self, messages: list[Message], output_model: type[T]
    ) -> tuple[T | None, str | None, str | None]:
        """Get structured output using OpenAI's beta parse endpoint synchronously.

        Args:
            messages: List of messages to send to the model.
            output_model: Pydantic model class to parse the response into.

        Returns:
            Tuple of (parsed_result, error_message, raw_response).
        """
        res: T | None = None
        msg: str | None = None
        raw: str | None = None

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                response_format=output_model,
                **self.generation_params,
            )
            raw = response.choices[0].message.content
            res = response.choices[0].message.parsed
        except (ValidationError, ValueError) as e:
            msg = f"Failed to obtain structured output for model {self.model_name} and messages {messages}: {e!s}"
            logger.warning(msg)
        else:
            if res is None:
                msg = "For some reason output wasn't parsed."
                logger.warning(msg)

        return res, msg, raw

    def _get_structured_output_vllm_sync(
        self, messages: list[Message], output_model: type[T]
    ) -> tuple[T | None, str | None, str | None]:
        """https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html."""
        res: T | None = None
        msg: str | None = None
        raw: str | None = None

        try:
            json_schema = output_model.model_json_schema()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                extra_body={"guided_json": json_schema},
                **self.generation_params,
            )
            raw = response.choices[0].message.content
            res = output_model.model_validate_json(raw)  # type: ignore[arg-type]
        except (ValidationError, ValueError) as e:
            msg = f"Failed to obtain structured output for model {self.model_name} and messages {messages}: {e!s}"
            logger.warning(msg)

        return res, msg, raw

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
        # Check cache first
        cached_result = self.cache.get(messages, output_model, backend, self.generation_params)
        if cached_result is not None:
            return cached_result

        current_messages = messages.copy()
        res: T | None = None

        for _ in range(max_retries + 1):
            if backend == "openai":
                res, error, raw = self._get_structured_output_openai_sync(current_messages, output_model)
            elif backend == "vllm":
                res, error, raw = self._get_structured_output_vllm_sync(current_messages, output_model)
            else:
                assert_never(backend)

            if res is not None:
                break

            if error is None:
                msg = "Structured output returned None but no error was caught."
                logger.exception(msg)
                raise RuntimeError(msg)

            current_messages.extend(self._create_retry_messages(error, raw))

        if res is None:
            msg = (
                f"Failed to generate valid structured output after {max_retries + 1} attempts.\n"
                f"Messages: {current_messages}"
            )
            logger.exception(msg)
            raise RuntimeError(msg)

        # Cache the successful result
        self.cache.set(messages, output_model, backend, self.generation_params, res)

        return res
