"""Schemas that are useful for working with prompts."""

from typing import TypedDict


class Message(TypedDict):
    """Schema for message to LLM."""

    role: str
    content: str


class Role:
    """Roles in a chat with LLM."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
