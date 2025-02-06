"""Chat template for evolution augmentation via concretizing."""

from typing import ClassVar

from autointent.generation.utterances.schemas import Message, Role
from autointent.schemas import Intent

from .base import EvolutionChatTemplate


class ConcreteEvolution(EvolutionChatTemplate):
    """Chat template for evolution augmentation via concretizing."""

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
                "Intent name: Reserve Restaurant\n"
                "Utterance: I want to make a reservation for dinner tonight."
            ),
        ),
        Message(role=Role.ASSISTANT, content="I want to reserve a table for 4 persons at 9 pm."),
        Message(
            role=Role.USER,
            content=(
                "Intent name: requesting technical support\n"
                "Utterance: I'm having trouble with my laptop."
            ),
        ),
        Message(role=Role.ASSISTANT, content="My laptop is constantly rebooting and overheating."),
    ]

    def __call__(self, utterance: str, intent_data: Intent) -> list[Message]:
        """Make chat to complete."""
        return [
            *self._messages,
            Message(role=Role.USER, content=f"Intent name: {intent_data.name or ''}\nUtterance: {utterance}"),
        ]
