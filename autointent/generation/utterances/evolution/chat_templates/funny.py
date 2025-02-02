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
                "{base_instruction}\n"
                "1. Rewrite the utterance in a humorous way while maintaining its original meaning.\n"
                "2. Use wordplay, exaggeration, or lighthearted phrasing.\n"
                "3. The rewritten utterance should still be understandable and relevant.\n"
                "4. Keep it within 15 words.\n\n"
                "Intent Class:\n"
                "Reserve Restaurant\n\n"
                "Utterance:\n"
                "I want to reserve a table for 4 persons at 9 pm."
            ),
        ),
        Message(role=Role.ASSISTANT, content="Gotta feed my squad at 9 pm. Got a table for us?"),
        Message(
            role=Role.USER,
            content=(
                "Intent Class:\n"
                "requesting technical support\n\n"
                "Utterance:\n"
                "My Lenovo laptop is constantly rebooting and overheating."
            ),
        ),
        Message(role=Role.ASSISTANT, content="My Lenovo thinks it's a phoenix—keeps dying and rising in flames."),
        Message(
            role=Role.USER,
            content=("Intent Class:\n" "{intent_name}\n\n" "Utterance:\n" "{utterance}"),
        ),
    ]

    def __call__(self, utterance: str, intent_data: Intent) -> list[Message]:
        """Generate chat for humorous tone adaptation."""
        return [
            *self._messages,
            Message(role=Role.USER, content=f"Intent Class:\n{intent_data.name or ''}\n\nUtterance:\n{utterance}"),
        ]
