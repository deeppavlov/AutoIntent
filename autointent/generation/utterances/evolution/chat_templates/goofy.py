"""Chat template for goofy tone augmentation."""

from typing import ClassVar

from autointent.generation.utterances.schemas import Message, Role
from autointent.schemas import Intent

from .base import EvolutionChatTemplate


class GoofyEvolution(EvolutionChatTemplate):
    """Chat template for goofy tone augmentation."""

    _messages: ClassVar[list[Message]] = [
        Message(
            role=Role.USER,
            content=(
                "{base_instruction}\n"
                "1. Rewrite the utterance in a silly, exaggerated, or nonsensical way while keeping the intent clear.\n"
                "2. Use playful words, randomness, or exaggeration.\n"
                "3. The rewritten utterance should still be answerable.\n"
                "4. Keep it under 15 words.\n\n"
                "Intent Class:\n"
                "Reserve Restaurant\n\n"
                "Utterance:\n"
                "I want to reserve a table for 4 persons at 9 pm."
            ),
        ),
        Message(role=Role.ASSISTANT, content="Need a feast for my hungry goblins at 9. Got room?"),
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
            role=Role.ASSISTANT, content="My laptop's having an existential crisis—keeps rebooting and melting. Help!"
        ),
        Message(
            role=Role.USER,
            content=("Intent Class:\n" "{intent_name}\n\n" "Utterance:\n" "{utterance}"),
        ),
    ]

    def __call__(self, utterance: str, intent_data: Intent) -> list[Message]:
        """Generate chat for goofy tone adaptation."""
        return [
            *self._messages,
            Message(role=Role.USER, content=f"Intent Class:\n{intent_data.name or ''}\n\nUtterance:\n{utterance}"),
        ]
