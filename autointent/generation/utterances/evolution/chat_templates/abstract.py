"""Chat template for evolution augmentation via abstractization."""

from typing import ClassVar

from autointent.generation.utterances.schemas import Message, Role
from autointent.schemas import Intent

from .base import EvolutionChatTemplate


class AbstractEvolution(EvolutionChatTemplate):
    """Chat template for evolution augmentation via abstraction."""

    _messages: ClassVar[list[Message]] = [
        Message(
            role=Role.USER,
            content=(
                "I want you to act as a rewriter. "
                "You will be provided with an utterance and the topic (name of intent class) of the utterance. "
                "You need to complicate the utterance using the following method:\n"
                "1. Rewrite the utterance by removing specific inquiries or replacing with more generic.\n"
                "2. Rewritten utterance should be concise and understandable by humans.\n"
                "3. Rewritten utterance must be fully answerable.\n"
                "4. Rewritten utterance should not contain more than 10 words.\n\n"
                "Intent name: Reserve Restaurant"
                "Utterance: I want to reserve a table for 4 persons at 9 pm."
            ),
        ),
        Message(role=Role.ASSISTANT, content="Please, reserve a table for me."),
        Message(
            role=Role.ASSISTANT,
            content=(
                "Intent name: requesting technical support"
                "Utterance: My Lenovo laptop is constantly rebooting and overheating."
            ),
        ),
        Message(role=Role.ASSISTANT, content="I'm having trouble with my laptop."),
    ]

    def __call__(self, utterance: str, intent_data: Intent) -> list[Message]:
        """Make chat to complete."""
        return [
            *self._messages,
            Message(role=Role.USER, content=f"Intent name: {intent_data.name or ''}\nUtterance: {utterance}"),
        ]
