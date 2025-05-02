"""Chat template for evolution augmentation via concretizing."""

from typing import ClassVar

from ._base_evolver import EvolutionChatTemplate
from ._evolution_templates_schemas import Message, Role


class ConcreteEvolution(EvolutionChatTemplate):
    """Chat template for evolution augmentation via concretizing."""

    name = "concrete"

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
            content="Intent name: requesting technical support\nUtterance: I'm having trouble with my laptop.",
        ),
        Message(role=Role.ASSISTANT, content="My laptop is constantly rebooting and overheating."),
    ]
