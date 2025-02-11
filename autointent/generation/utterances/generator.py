"""Wrapper class for accessing OpenAI API."""

import os

import openai
from dotenv import load_dotenv

from .schemas import Message


class Generator:
    """Wrapper class for accessing OpenAI API."""

    def __init__(self) -> None:
        """Initialize."""
        load_dotenv()
        self.client = openai.OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
        self.async_client = openai.AsyncOpenAI(
            base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"]
        )
        self.model_name = os.environ["OPENAI_MODEL_NAME"]

    def get_chat_completion(self, messages: list[Message]) -> str:
        """Prompt LLM and return its answer synchronously."""
        response = self.client.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]
            model=self.model_name,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content  # type: ignore[return-value]

    async def get_chat_completion_async(self, messages: list[Message]) -> str:
        """Prompt LLM and return its answer asynchronously."""
        response = await self.async_client.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]
            model=self.model_name,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content  # type: ignore[return-value]
