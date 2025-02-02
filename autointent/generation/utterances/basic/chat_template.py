"""Chat template for evolution augmentation via abstractization."""

import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import ClassVar

from autointent import Dataset
from autointent.generation.utterances.schemas import Message, Role
from autointent.schemas import Intent


class BaseSynthesizer(ABC):
    """Base class."""

    @abstractmethod
    def __call__(self, intent_data: Intent, n_examples: int) -> list[Message]:
        """Generate examples for this intent."""


class SynthesizerChatTemplate(BaseSynthesizer):
    """Chat template for generating additional examples for a given intent class."""

    __messages: ClassVar[list[Message]] = [
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
            content=("1. Can you tell me the forecast for tomorrow?\n" "2. Is it going to rain this weekend?"),
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

    def __init__(
        self,
        dataset: Dataset,
        split: str,
        extra_instructions: str | None = None,
        max_sample_utterances: int | None = None,
    ) -> None:
        """Initialize."""
        if extra_instructions is None:
            extra_instructions = ""

        self._messages = deepcopy(self.__messages)

        msg = self._messages[0]
        msg["content"] = msg["content"].format(extra_instructions=extra_instructions)

        self.dataset = dataset
        self.split = split
        self.max_sample_utterances = max_sample_utterances

    def __call__(self, intent_data: Intent, n_examples: int) -> list[Message]:
        """Generate additional examples for the provided intent class."""
        filtered_split = self.dataset[self.split].filter(lambda sample: sample[Dataset.label_feature] == intent_data.id)
        sample_utterances = filtered_split[Dataset.utterance_feature]
        if self.max_sample_utterances is not None:
            sample_utterances = random.sample(sample_utterances, k=self.max_sample_utterances)
        return [
            *self._messages,
            Message(
                role=Role.USER,
                content=f"Intent name: {intent_data.name}\n\n"
                f"Example Utterances:\n{sample_utterances}\n\n"
                f"Please generate {n_examples} more examples for the provided intent class.\n",
            ),
        ]
