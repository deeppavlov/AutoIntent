"""Chat template for evolution augmentation via reasoning."""

from typing import ClassVar

from ._base_evolver import EvolutionChatTemplate
from ._evolution_templates_schemas import Message, Role


class ReasoningEvolution(EvolutionChatTemplate):
    """Chat template for evolution augmentation via reasoning."""

    name = "reasoning"

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
                "Intent name: Asking for Directions\n"
                "Utterance: How do I get to the nearest coffee shop?"
            ),
        ),
        Message(role=Role.ASSISTANT, content="If there are some place where I can buy a coffee, how can I get there?"),
        Message(
            role=Role.USER,
            content=(
                "Intent name: requesting technical support\n"
                "Utterance: I want to get help from technical support for my laptop."
            ),
        ),
        Message(role=Role.ASSISTANT, content="I don't know what's happening with my laptop."),
    ]
