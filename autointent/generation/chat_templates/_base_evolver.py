"""Base class for chat templates for evolution augmentation."""

from typing import ClassVar

from autointent.schemas import Intent

from ._evolution_templates_schemas import Message, Role


class EvolutionChatTemplate:
    """Base class for chat templates for evolution augmentation."""

    _messages: ClassVar[list[Message]]
    name: str

    def __call__(self, utterance: str, intent_data: Intent) -> list[Message]:
        """Generate a list of messages to request additional examples for the given intent.

        Args:
            utterance: Utterance to be used for generation.
            intent_data: Intent data for which to generate examples.

        Returns:
            List of messages for the chat template.
        """
        invoke_message = Message(
            role=Role.USER,
            content=f"Intent name: {intent_data.name or ''}\nUtterance: {utterance}",
        )
        return [*self._messages, invoke_message]
