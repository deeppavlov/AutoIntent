"""Chat template for informal tone augmentation."""

from typing import ClassVar

from autointent.generation.utterances.schemas import Message, Role
from autointent.schemas import Intent

from .base import EvolutionChatTemplate


class InformalEvolution(EvolutionChatTemplate):
    """Chat template for informal tone augmentation."""

    _messages: ClassVar[list[Message]] = [
        Message(
            role=Role.USER,
            content=(
                "{base_instruction}\n"
                "1. Rewrite the utterance in a more casual and relaxed tone.\n"
                "2. Use contractions, friendly language, and a conversational style.\n"
                "3. The rewritten utterance should feel natural in an informal conversation.\n"
                "4. Keep it under 15 words.\n\n"
                "Intent Class:\n"
                "Reserve Restaurant\n\n"
                "Utterance:\n"
                "I want to reserve a table for 4 persons at 9 pm."
            ),
        ),
        Message(role=Role.ASSISTANT, content="Hey, can I book a table for 4 at 9?"),
        Message(
            role=Role.USER,
            content=(
                "Intent Class:\n"
                "requesting technical support\n\n"
                "Utterance:\n"
                "My Lenovo laptop is constantly rebooting and overheating."
            ),
        ),
        Message(role=Role.ASSISTANT, content="My Lenovo keeps crashing and getting super hot. Any ideas?"),
        Message(
            role=Role.USER,
            content=("Intent Class:\n" "{intent_name}\n\n" "Utterance:\n" "{utterance}"),
        ),
    ]

    def __call__(self, utterance: str, intent_data: Intent) -> list[Message]:
        """Generate chat for informal tone adaptation."""
        return [
            *self._messages,
            Message(role=Role.USER, content=f"Intent Class:\n{intent_data.name or ''}\n\nUtterance:\n{utterance}"),
        ]
