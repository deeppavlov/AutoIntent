"""Chat template for formal tone augmentation."""

from typing import ClassVar

from ._base_evolver import EvolutionChatTemplate
from ._evolution_templates_schemas import Message, Role


class FormalEvolution(EvolutionChatTemplate):
    """Chat template for formal tone augmentation."""

    name: str = "formal"

    _messages: ClassVar[list[Message]] = [
        Message(
            role=Role.USER,
            content=(
                "I want you to act as a rewriter. "
                "You will be provided with an utterance and the topic (name of intent class) of the utterance. "
                "You need to rewrite the utterance in a more formal tone using the following method:\n"
                "1. Rewrite the utterance in a more formal tone.\n"
                "2. Use polite and professional language while maintaining clarity.\n"
                "3. The rewritten utterance should be grammatically correct and complete.\n"
                "4. Keep the rewritten utterance within 15 words.\n\n"
                "Intent name: Reserve Restaurant"
                "Utterance: I want to reserve a table for 4 persons at 9 pm."
            ),
        ),
        Message(role=Role.ASSISTANT, content="I would like to make a reservation for four guests at 9 pm."),
        Message(
            role=Role.ASSISTANT,
            content=(
                "Intent name: requesting technical support\n"
                "Utterance: My Lenovo laptop is constantly rebooting and overheating."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content="My Lenovo laptop frequently restarts and experiences overheating issues. Kindly assist.",
        ),
    ]
