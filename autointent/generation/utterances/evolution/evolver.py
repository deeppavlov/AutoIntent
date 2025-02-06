"""
Evolutionary strategy to augmenting utterances.

Deeply inspired by DeepEval evolutions.
"""

import asyncio
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
        self,
        generator: Generator,
        prompt_makers: Sequence[Callable[[str, Intent], list[Message]]],
        seed: int = 0,
        async_mode: bool = False
    ) -> None:
        """Initialize."""
        self.generator = generator
        self.prompt_makers = prompt_makers
        self.async_mode = async_mode
        random.seed(seed)

    def _evolve(self, utterance: str, intent_data: Intent) -> str:
        """Apply evolutions single time synchronously."""
        maker = random.choice(self.prompt_makers)
        chat = maker(utterance, intent_data)
        return self.generator.get_chat_completion(chat)

    async def _evolve_async(self, utterance: str, intent_data: Intent) -> str:
        """Apply evolutions single time asynchronously."""
        maker = random.choice(self.prompt_makers)
        chat = maker(utterance, intent_data)
        return await self.generator.get_chat_completion_async(chat)

    def __call__(self, utterance: str, intent_data: Intent, n_evolutions: int = 1) -> list[str]:
        """Apply evolutions multiple times (synchronously)."""
        return [self._evolve(utterance, intent_data) for _ in range(n_evolutions)]

    async def _call_async(self, utterance: str, intent_data: Intent, n_evolutions: int = 1) -> list[str]:
        """Apply evolutions multiple times asynchronously."""
        tasks = [self._evolve_async(utterance, intent_data) for _ in range(n_evolutions)]
        return await asyncio.gather(*tasks)

    def augment(
        self,
        dataset: Dataset,
        split_name: str = Split.TRAIN,
        n_evolutions: int = 1,
        update_split: bool = True,
        batch_size: int = 4
    ) -> list[Sample]:
        """
        Augment some split of dataset.

        Note that for now it supports only single-label datasets.
        """
        if self.async_mode:
            return asyncio.get_event_loop().run_until_complete(
                self._augment_async(
                    dataset=dataset,
                    split_name=split_name,
                    n_evolutions=n_evolutions,
                    update_split=update_split,
                    batch_size=batch_size
                )
            )

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

    async def _augment_async(
        self,
        dataset: Dataset,
        split_name: str = Split.TRAIN,
        n_evolutions: int = 1,
        update_split: bool = True,
        batch_size: int = 4
    ) -> list[Sample]:
        original_split = dataset[split_name]
        new_samples = []

        total_samples = len(original_split)
        for start_idx in range(0, total_samples, batch_size):
            batch = original_split[start_idx : start_idx + batch_size]
            tasks = []
            for utterance, label in zip(
                batch[Dataset.utterance_feature],
                batch[Dataset.label_feature],
                strict=False
            ):
                intent_data = next(intent for intent in dataset.intents if intent.id == label)
                tasks.append(
                    self._call_async(utterance=utterance, intent_data=intent_data, n_evolutions=n_evolutions)
                )

            batch_results = await asyncio.gather(*tasks)

            for i, generated_utterances in enumerate(batch_results):
                intent_data = next(
                    intent for intent in dataset.intents if intent.id == batch[Dataset.label_feature][i]
                )
                new_samples.extend(
                    [{Dataset.label_feature: intent_data.id, Dataset.utterance_feature: ut}
                        for ut in generated_utterances]
                )

        if update_split:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])

        return [Sample(**sample) for sample in new_samples]
