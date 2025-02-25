"""Base class for chat templates for evolution augmentation."""

from typing import ClassVar

from autointent.generation.utterances.schemas import Message, Role
from autointent.schemas import Intent


class EvolutionChatTemplate:
    """Base class for chat templates for evolution augmentation."""

    _messages: ClassVar[list[Message]]
    name: str

    def __call__(self, utterance: str, intent_data: Intent) -> list[Message]:
        """Make a chat to complete by LLM."""
        invoke_message = Message(
            role=Role.USER,
            content=f"Intent name: {intent_data.name or ''}\nUtterance: {utterance}",
        )
        return [*self._messages, invoke_message]
