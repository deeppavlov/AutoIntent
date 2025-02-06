"""Chat template for humorous tone augmentation."""

from typing import ClassVar

from autointent.generation.utterances.schemas import Message, Role
from autointent.schemas import Intent

from .base import EvolutionChatTemplate


class FunnyEvolution(EvolutionChatTemplate):
    """Chat template for humorous tone augmentation."""

    _messages: ClassVar[list[Message]] = [
        Message(
            role=Role.USER,
            content=(
                "I want you to act as a rewriter. "
                "You will be provided with an utterance and the topic (name of intent class) of the utterance. "
                "You need to rewrite the utterance in a humorous way while maintaining its original meaning using "
                "the following method:\n"
                "1. Rewrite the utterance in a humorous way while maintaining its original meaning.\n"
                "2. Use wordplay, exaggeration, or lighthearted phrasing.\n"
                "3. The rewritten utterance should still be understandable and relevant.\n"
                "4. Keep it within 15 words.\n\n"
                "Intent name: Reserve Restaurant"
                "Utterance: I want to reserve a table for 4 persons at 9 pm."
            ),
        ),
        Message(role=Role.ASSISTANT, content="Gotta feed my squad at 9 pm. Got a table for us?"),
        Message(
            role=Role.USER,
            content=(
                "Intent name: requesting technical support\n"
                "Utterance: My Lenovo laptop is constantly rebooting and overheating."
            ),
        ),
        Message(role=Role.ASSISTANT, content="My Lenovo thinks it's a phoenix—keeps dying and rising in flames."),
    ]

    def __call__(self, utterance: str, intent_data: Intent) -> list[Message]:
        """Generate chat for humorous tone adaptation."""
        return [
            *self._messages,
            Message(role=Role.USER, content=f"Intent name: {intent_data.name or ''}\nUtterance: {utterance}"),
        ]
