"""
Evolutionary strategy to augmenting utterances.

Deeply inspired by DeepEval evolutions.
"""

import asyncio
import random
from collections.abc import Sequence

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation.utterances.evolution.chat_templates import EvolutionChatTemplate
from autointent.generation.utterances.generator import Generator
from autointent.schemas import Intent


class UtteranceEvolver:
    """
    Evolutionary strategy to augmenting utterances.

    Deeply inspired by DeepEval evolutions. This method takes single utterance and prompts LLM
    to change it in a specific way.
    """

    def __init__(
        self,
        generator: Generator,
        prompt_makers: Sequence[EvolutionChatTemplate],
        seed: int = 0,
        async_mode: bool = False,
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
        """Apply evolutions a single time (asynchronously)."""
        maker = random.choice(self.prompt_makers)
        chat = maker(utterance, intent_data)
        return await self.generator.get_chat_completion_async(chat)

    def __call__(self, utterance: str, intent_data: Intent, n_evolutions: int = 1) -> list[str]:
        """Apply evolutions multiple times (synchronously)."""
        return [self._evolve(utterance, intent_data) for _ in range(n_evolutions)]

    def augment(
        self,
        dataset: Dataset,
        split_name: str = Split.TRAIN,
        n_evolutions: int = 1,
        update_split: bool = True,
        batch_size: int = 4,
    ) -> HFDataset:
        """
        Augment some split of dataset.

        Note that for now it supports only single-label datasets.
        """
        if self.async_mode:
            return asyncio.run(
                self._augment_async(
                    dataset=dataset,
                    split_name=split_name,
                    n_evolutions=n_evolutions,
                    update_split=update_split,
                    batch_size=batch_size,
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

        generated_split = HFDataset.from_list(new_samples)
        if update_split:
            dataset[split_name] = concatenate_datasets([original_split, generated_split])

        return generated_split

    async def _augment_async(
        self,
        dataset: Dataset,
        split_name: str = Split.TRAIN,
        n_evolutions: int = 1,
        update_split: bool = True,
        batch_size: int = 4,
    ) -> HFDataset:
        original_split = dataset[split_name]
        new_samples = []

        tasks = []
        labels = []
        for sample in original_split:
            utterance = sample[Dataset.utterance_feature]
            label = sample[Dataset.label_feature]
            intent_data = next(intent for intent in dataset.intents if intent.id == label)
            for _ in range(n_evolutions):
                tasks.append(self._evolve_async(utterance, intent_data))
                labels.append(intent_data.id)

        for start_idx in range(0, len(tasks), batch_size):
            batch_tasks = tasks[start_idx : start_idx + batch_size]
            batch_labels = labels[start_idx : start_idx + batch_size]
            batch_results = await asyncio.gather(*batch_tasks)
            for result, intent_id in zip(batch_results, batch_labels, strict=False):
                new_samples.append({Dataset.label_feature: intent_id, Dataset.utterance_feature: result})

        generated_split = HFDataset.from_list(new_samples)
        if update_split:
            dataset[split_name] = concatenate_datasets([original_split, generated_split])

        return generated_split
