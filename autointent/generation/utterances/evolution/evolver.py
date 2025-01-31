"""
Evolutionary strategy to augmenting utterances.

Deeply inspired by DeepEval evolutions.
"""

import random
from collections.abc import Callable, Sequence

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation.utterances.generator import Generator
from autointent.generation.utterances.schemas import Message
from autointent.schemas import Intent, Sample


class UtteranceEvolver:
    """
    Evolutionary strategy to augmenting utterances.

    Deeply inspired by DeepEval evolutions. This method takes single utterance and prompts LLM
    to change it in a specific way.
    """

    def __init__(
        self, generator: Generator, prompt_makers: Sequence[Callable[[str, Intent], list[Message]]], seed: int = 0
    ) -> None:
        """Initialize."""
        self.generator = generator
        self.prompt_makers = prompt_makers
        random.seed(seed)

    def _evolve(self, utterance: str, intent_data: Intent) -> str:
        """Apply evolutions single time."""
        maker = random.choice(self.prompt_makers)
        chat = maker(utterance, intent_data)
        return self.generator.get_chat_completion(chat)

    def __call__(self, utterance: str, intent_data: Intent, n_evolutions: int = 1) -> list[str]:
        """Apply evolutions mupltiple times."""
        return [self._evolve(utterance, intent_data) for _ in range(n_evolutions)]

    def augment(
        self, dataset: Dataset, split_name: str = Split.TRAIN, n_evolutions: int = 1, update_split: bool = True
    ) -> list[Sample]:
        """
        Augment some split of dataset.

        Note that for now it supports only single-label datasets.
        """
        original_split = dataset[split_name]
        new_samples = []
        for sample in original_split:
            utterance = sample[Dataset.utterance_feature]
            label = sample[Dataset.label_feature]
            intent_data = next(intent for intent in dataset.intents if intent.id == label)
            generated_utterances = self(utterance=utterance, intent_data=intent_data, n_evolutions=n_evolutions)
            new_samples.extend(
                [{Dataset.label_feature: intent_data.id, Dataset.utterance_feature: ut} for ut in generated_utterances]
            )
        if update_split:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])
        return [Sample(**sample) for sample in new_samples]
