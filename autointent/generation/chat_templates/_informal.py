"""Chat template for informal tone augmentation."""

from typing import ClassVar

from ._base_evolver import EvolutionChatTemplate
from ._evolution_templates_schemas import Message, Role


class InformalEvolution(EvolutionChatTemplate):
    """Chat template for informal tone augmentation."""

    name: str = "informal"
    _messages: ClassVar[list[Message]] = [
        Message(
            role=Role.USER,
            content=(
                "I want you to act as a rewriter. "
                "You will be provided with an utterance and the topic (name of intent class) of the utterance. "
                "You need to rewrite the utterance in a more casual and relaxed tone using the following method:\n"
                "1. Rewrite the utterance in a more casual and relaxed tone.\n"
                "2. Use contractions, friendly language, and a conversational style.\n"
                "3. The rewritten utterance should feel natural in an informal conversation.\n"
                "4. Keep it under 15 words.\n\n"
                "Intent name: Reserve Restaurant"
                "Utterance: I want to reserve a table for 4 persons at 9 pm."
            ),
        ),
        Message(role=Role.ASSISTANT, content="Hey, can I book a table for 4 at 9?"),
        Message(
            role=Role.USER,
            content=(
                "Intent name: requesting technical support\n"
                "Utterance: My Lenovo laptop is constantly rebooting and overheating."
            ),
        ),
        Message(role=Role.ASSISTANT, content="My Lenovo keeps crashing and getting super hot. Any ideas?"),
    ]
