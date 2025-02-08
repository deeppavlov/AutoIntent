"""Basic generation of new utterances from existing ones."""

import asyncio
from collections.abc import Callable

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation.utterances.generator import Generator
from autointent.generation.utterances.schemas import Message
from autointent.schemas import Intent, Sample


class UtteranceGenerator:
    """
    Basic generation of new utterances from existing ones.

    This augmentation method simply prompts LLM to look at existing examples
    and generate similar. Additionally, it can consider some aspects of style,
    punctuation, and length of the desired generations.
    """

    def __init__(
        self, generator: Generator, prompt_maker: Callable[[Intent, int], list[Message]], async_mode: bool = False
    ) -> None:
        """Initialize."""
        self.generator = generator
        self.prompt_maker = prompt_maker
        self.async_mode = async_mode

    def __call__(self, intent_data: Intent, n_generations: int) -> list[str]:
        """Generate new utterances."""
        messages = self.prompt_maker(intent_data, n_generations)
        response_text = self.generator.get_chat_completion(messages)
        return _extract_utterances(response_text)

    async def _call_async(self, intent_data: Intent, n_generations: int) -> list[str]:
        """Generate new utterances asynchronously."""
        messages = self.prompt_maker(intent_data, n_generations)
        response_text = await self.generator.get_chat_completion_async(messages)
        return _extract_utterances(response_text)

    def augment(
        self,
        dataset: Dataset,
        split_name: str = Split.TRAIN,
        n_generations: int = 5,
        update_split: bool = True,
        batch_size: int = 4,
    ) -> list[Sample]:
        """
        Augment some split of dataset.

        :param dataset: Dataset object
        :param split_name: Dataset split (default is TRAIN)
        :param n_generations: Number of utterances to generate per intent
        :param update_split: Whether to update the dataset split
        :param batch_size: Batch size for async generation
        :return: List of generated samples
        """
        if self.async_mode:
            return asyncio.run(self._augment_async(dataset, split_name, n_generations, update_split, batch_size))

        original_split = dataset[split_name]
        new_samples = []
        for intent in dataset.intents:
            generated_utterances = self(intent_data=intent, n_generations=n_generations)
            new_samples.extend(
                [{Dataset.label_feature: intent.id, Dataset.utterance_feature: ut} for ut in generated_utterances]
            )

        if update_split:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])

        return [Sample(**sample) for sample in new_samples]

    async def _augment_async(
        self,
        dataset: Dataset,
        split_name: str = Split.TRAIN,
        n_generations: int = 5,
        update_split: bool = True,
        batch_size: int = 4,
    ) -> list[Sample]:
        """
        Augment some split of dataset asynchronously in batches.

        :param dataset: Dataset object
        :param split_name: Dataset split (default is TRAIN)
        :param n_generations: Number of utterances to generate per intent
        :param update_split: Whether to update the dataset split
        :param batch_size: Batch size for async generation
        :return: List of generated samples
        """
        original_split = dataset[split_name]
        new_samples = []

        results = []
        for start_idx in range(0, len(dataset.intents), batch_size):
            batch_intents = dataset.intents[start_idx : start_idx + batch_size]
            tasks = [self._call_async(intent_data=intent, n_generations=n_generations) for intent in batch_intents]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        for i, generated_utterances in enumerate(results):
            intent = dataset.intents[i]
            new_samples.extend(
                [{Dataset.label_feature: intent.id, Dataset.utterance_feature: ut} for ut in generated_utterances]
            )

        if update_split:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])

        return [Sample(**sample) for sample in new_samples]


def _extract_utterances(response_text: str) -> list[str]:
    """
    Parse LLM output.

    Inverse function to :py:func:`_format_utterances`.
    """
    raw_utterances = response_text.split("\n")
    # remove enumeration
    return [ut[ut.find(" ") + 1 :] if " " in ut else ut for ut in raw_utterances]
