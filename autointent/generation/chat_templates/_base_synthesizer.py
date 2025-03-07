"""Base class for chat template for class-wise augmentation."""

import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import ClassVar

from autointent import Dataset
from autointent.custom_types import Split
from autointent.schemas import Intent

from ._evolution_templates_schemas import Message, Role


class BaseChatTemplate(ABC):
    """Base class."""

    @abstractmethod
    def __call__(self, intent_data: Intent, n_examples: int) -> list[Message]:
        """Generate a list of messages to request additional examples for the given intent.

        Args:
            intent_data: Intent data for which to generate examples.
            n_examples: Number of examples to generate.

        Returns:
            List of messages for the chat template.
        """


class BaseSynthesizerTemplate(BaseChatTemplate):
    """Base chat template for generating additional examples for a given intent."""

    _MESSAGES_TEMPLATE: ClassVar[list[Message]]
    _INTENT_NAME_LABEL: ClassVar[str]
    _EXAMPLE_UTTERANCES_LABEL: ClassVar[str]
    _GENERATE_INSTRUCTION: ClassVar[str]

    def __init__(
        self,
        dataset: Dataset,
        split: str = Split.TRAIN,
        extra_instructions: str | None = None,
        max_sample_utterances: int | None = None,
    ) -> None:
        """Initialize the BaseSynthesizerTemplate.

        Args:
            dataset: Dataset to use for generating examples.
            split: Dataset split to use for generating examples.
            extra_instructions: Additional instructions for the model.
            max_sample_utterances: Maximum number of sample utterances to include.

        Raises:
            ValueError: If the dataset is not provided.
        """
        if extra_instructions is None:
            extra_instructions = ""

        self._messages = deepcopy(self._MESSAGES_TEMPLATE)

        if self._messages:
            self._messages[0]["content"] = self._messages[0]["content"].format(extra_instructions=extra_instructions)

        self.dataset = dataset
        self.split = split
        self.max_sample_utterances = max_sample_utterances

    def __call__(self, intent_data: Intent, n_examples: int) -> list[Message]:
        """Generate a list of messages to request additional examples for the given intent.

        Args:
            intent_data: Intent data for which to generate examples.
            n_examples: Number of examples to generate.

        Returns:
            List of messages for the chat template.
        """
        in_domain_samples = self.dataset[self.split].filter(lambda sample: sample[Dataset.label_feature] is not None)
        if self.dataset.multilabel:
            filter_fn = lambda sample: sample[Dataset.label_feature][intent_data.id] == 1  # noqa: E731
        else:
            filter_fn = lambda sample: sample[Dataset.label_feature] == intent_data.id  # noqa: E731

        filtered_split = in_domain_samples.filter(filter_fn)
        sample_utterances = filtered_split[Dataset.utterance_feature]

        if self.max_sample_utterances is not None and len(sample_utterances) > self.max_sample_utterances:
            sample_utterances = random.sample(sample_utterances, k=self.max_sample_utterances)

        return [
            *self._messages,
            self._create_final_message(intent_data, n_examples, sample_utterances),
        ]

    def _create_final_message(self, intent_data: Intent, n_examples: int, sample_utterances: list[str]) -> Message:
        """Create the final message for the chat template.

        Args:
            intent_data: Intent data for which to generate examples.
            n_examples: Number of examples to generate.
            sample_utterances: Sample utterances to include.

        Returns:
            The final message for the chat template.
        """
        content = f"{self._INTENT_NAME_LABEL}: {intent_data.name}\n\n{self._EXAMPLE_UTTERANCES_LABEL}:\n"

        if sample_utterances:
            numbered_utterances = "\n".join(f"{i + 1}. {utt}" for i, utt in enumerate(sample_utterances))
            content += numbered_utterances + "\n\n"

        content += self._GENERATE_INSTRUCTION.format(n_examples=n_examples)
        return Message(role=Role.USER, content=content)
