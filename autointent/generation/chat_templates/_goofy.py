"""Chat template for goofy tone augmentation."""

from typing import ClassVar

from ._base_evolver import EvolutionChatTemplate
from ._evolution_templates_schemas import Message, Role


class GoofyEvolution(EvolutionChatTemplate):
    """Chat template for goofy tone augmentation."""

    name: str = "goofy"
    _messages: ClassVar[list[Message]] = [
        Message(
            role=Role.USER,
            content=(
                "I want you to act as a rewriter. "
                "You will be provided with an utterance and the topic (name of intent class) of the utterance. "
                "You need to rewrite the utterance in a silly, exaggerated, or nonsensical way while keeping "
                "the intent clear using the following method:\n"
                "1. Rewrite the utterance in a silly, exaggerated, or nonsensical way while keeping the intent clear.\n"
                "2. Use playful words, randomness, or exaggeration.\n"
                "3. The rewritten utterance should still be answerable.\n"
                "4. Keep it under 15 words.\n\n"
                "Intent name: Reserve Restaurant"
                "Utterance: I want to reserve a table for 4 persons at 9 pm."
            ),
        ),
        Message(role=Role.ASSISTANT, content="Need a feast for my hungry goblins at 9. Got room?"),
        Message(
            role=Role.USER,
            content=(
                "Intent name: requesting technical support\n"
                "Utterance: My Lenovo laptop is constantly rebooting and overheating."
            ),
        ),
        Message(
            role=Role.ASSISTANT, content="My laptop's having an existential crisis—keeps rebooting and melting. Help!"
        ),
    ]
