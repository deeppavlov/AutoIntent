"""Chat template for formal tone augmentation."""

from typing import ClassVar

from autointent.generation.utterances.schemas import Message, Role
from autointent.schemas import Intent

from .base import EvolutionChatTemplate


class FormalEvolution(EvolutionChatTemplate):
    """Chat template for formal tone augmentation."""

    _messages: ClassVar[list[Message]] = [
        Message(
            role=Role.USER,
            content=(
                "{base_instruction}\n"
                "1. Rewrite the utterance in a more formal tone.\n"
                "2. Use polite and professional language while maintaining clarity.\n"
                "3. The rewritten utterance should be grammatically correct and complete.\n"
                "4. Keep the rewritten utterance within 15 words.\n\n"
                "Intent Class:\n"
                "Reserve Restaurant\n\n"
                "Utterance:\n"
                "I want to reserve a table for 4 persons at 9 pm."
            ),
        ),
        Message(role=Role.ASSISTANT, content="I would like to make a reservation for four guests at 9 pm."),
        Message(
            role=Role.USER,
            content=(
                "Intent Class:\n"
                "requesting technical support\n\n"
                "Utterance:\n"
                "My Lenovo laptop is constantly rebooting and overheating."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content="My Lenovo laptop frequently restarts and experiences overheating issues. Kindly assist.",
        ),
        Message(
            role=Role.USER,
            content=("Intent Class:\n" "{intent_name}\n\n" "Utterance:\n" "{utterance}"),
        ),
    ]

    def __call__(self, utterance: str, intent_data: Intent) -> list[Message]:
        """Generate chat for formal tone adaptation."""
        return [
            *self._messages,
            Message(role=Role.USER, content=f"Intent Class:\n{intent_data.name or ''}\n\nUtterance:\n{utterance}"),
        ]
