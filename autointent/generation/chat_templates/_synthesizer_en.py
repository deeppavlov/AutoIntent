"""Chat template for class-wise augmentation for English datasets."""

from typing import ClassVar

from ._base_synthesizer import BaseSynthesizerTemplate
from ._evolution_templates_schemas import Message, Role


class EnglishSynthesizerTemplate(BaseSynthesizerTemplate):
    """Chat template for generating additional examples for a given intent class."""

    _INTENT_NAME_LABEL: ClassVar[str] = "Intent name"
    _EXAMPLE_UTTERANCES_LABEL: ClassVar[str] = "Example Utterances"
    _GENERATE_INSTRUCTION: ClassVar[str] = "Please generate {n_examples} more examples for the provided intent class.\n"

    _MESSAGES_TEMPLATE: ClassVar[list[Message]] = [
        Message(
            role=Role.USER,
            content=(
                "You will be provided with a set of example utterances and the name "
                "of the common topic (intent name) of these utterances. "
                "Your task is to generate more examples that fit within the same intent name.\n\n"
                "Note:\n"
                "- You can generate similar utterances with only slot values changed\n"
                "- You can generate completely different utterance from the same intent name\n"
                "- Intent name can be missed, then you should infer from example utterances only\n"
                "- Example utterances can be missed, then you should infer from intent name only\n"
                "{extra_instructions}\n\n"
                "Intent name: ordering_pizza\n\n"
                "Example Utterances:\n"
                "1. I want to order a large pepperoni pizza.\n"
                "2. Can I get a medium cheese pizza with extra olives?\n"
                "3. Please deliver a small veggie pizza to my address.\n\n"
                "Please generate 3 more examples for the provided intent name."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "1. I'd like to order a large margherita pizza.\n"
                "2. Can you deliver a medium Hawaiian pizza with extra pineapple?\n"
                "3. Please send a small BBQ chicken pizza to my home."
            ),
        ),
        Message(
            role=Role.USER,
            content=(
                "Intent name: booking a hotel\n\n"
                "Example Utterances:\n"
                "1. I need to book a room for two nights in New York.\n\n"
                "Please generate 2 more examples for the provided intent name."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "1. Can you reserve a deluxe room for my trip to Tokyo?\n"
                "2. I need to book a hotel room with a mountain view in Denver."
            ),
        ),
        Message(
            role=Role.USER,
            content=(
                "Intent name:\n\n"
                "Example Utterances:\n"
                "1. What is the weather like today?\n\n"
                "Please generate 2 more examples for the provided intent class."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content="1. Can you tell me the forecast for tomorrow?\n2. Is it going to rain this weekend?",
        ),
        Message(
            role=Role.USER,
            content=(
                "Intent name: Scheduling a Meeting\n\n"
                "Example Utterances:\n\n"
                "Please generate 3 more examples for the provided intent class."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "1. I need to schedule a meeting for next Tuesday.\n"
                "2. Can you set up a conference call for tomorrow afternoon?\n"
                "3. Please arrange a meeting with the marketing team next week."
            ),
        ),
    ]
