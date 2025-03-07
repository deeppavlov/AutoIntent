"""Evolutionary strategy to augmenting utterances.

Deeply inspired by DeepEval evolutions.
"""

import asyncio
import random
from collections.abc import Sequence

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation import Generator
from autointent.generation.chat_templates import EvolutionChatTemplate
from autointent.schemas import Intent


class UtteranceEvolver:
    """Evolutionary strategy to augmenting utterances.

    Deeply inspired by DeepEval evolutions. This method takes single utterance and prompts LLM
    to change it in a specific way.

    Args:
        generator: Generator instance for generating utterances.
        prompt_makers: List of prompt makers for generating prompts.
        seed: Random seed for reproducibility.
        async_mode: Whether to use asynchronous mode for generation.

    Usage
    -----

    .. code-block:: python

        from autointent import Dataset
        from autointent.generation import Generator
        from autointent.generation.utterances import UtteranceEvolver
        from autointent.generation.chat_templates import GoofyEvolution, InformalEvolution

        dataset = Dataset.from_json(path_to_json)
        generator = Generator()
        evolver = UtteranceEvolver(generator, prompt_makers=[GoofyEvolution(), InformalEvolution()])
        evolver.augment(dataset)

    """

    def __init__(
        self,
        generator: Generator,
        prompt_makers: Sequence[EvolutionChatTemplate],
        seed: int = 0,
        async_mode: bool = False,
    ) -> None:
        self.generator = generator
        self.prompt_makers = prompt_makers
        self.async_mode = async_mode
        random.seed(seed)

    def _evolve(self, utterance: str, intent_data: Intent) -> str:
        """Apply evolutions a single time.

        Args:
            utterance: Utterance to be evolved.
            intent_data: Intent data for which to evolve the utterance.
        """
        maker = random.choice(self.prompt_makers)
        chat = maker(utterance, intent_data)
        return self.generator.get_chat_completion(chat)

    async def _evolve_async(self, utterance: str, intent_data: Intent) -> str:
        """Apply evolutions a single time asynchronously.

        Args:
            utterance: Utterance to be evolved.
            intent_data: Intent data for which to evolve the utterance.
        """
        maker = random.choice(self.prompt_makers)
        chat = maker(utterance, intent_data)
        return await self.generator.get_chat_completion_async(chat)

    def __call__(
        self, utterance: str, intent_data: Intent, n_evolutions: int = 1, sequential: bool = False
    ) -> list[str]:
        """Apply evolutions to the utterance.

        Args:
            utterance: Utterance to be evolved.
            intent_data: Intent data for which to evolve the utterance.
            n_evolutions: Number of evolutions to apply.
            sequential: Whether to apply evolutions sequentially.
        """
        current_utterance = utterance
        generated_utterances = []

        for _ in range(n_evolutions):
            gen_utt = self._evolve(current_utterance, intent_data)
            generated_utterances.append(gen_utt)

            if sequential:
                current_utterance = gen_utt

        return generated_utterances

    def augment(
        self,
        dataset: Dataset,
        split_name: str = Split.TRAIN,
        n_evolutions: int = 1,
        update_split: bool = True,
        batch_size: int = 4,
        sequential: bool = False,
    ) -> HFDataset:
        """Add LLM-generated samples to some split of dataset.

        Args:
            dataset: Dataset object.
            split_name: Dataset split (default is TRAIN).
            n_evolutions: Number of evolutions to apply.
            update_split: Whether to update the dataset split.
            batch_size: Batch size for async generation.
            sequential: Whether to apply evolutions sequentially.
        """
        if self.async_mode:
            if sequential:
                error = "Sequential and async modes are not compatible"
                raise ValueError(error)

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
            generated_utterances = self(
                utterance=utterance, intent_data=intent_data, n_evolutions=n_evolutions, sequential=sequential
            )
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
        """Augment some split of dataset asynchronously.

        Args:
            dataset: Dataset object.
            split_name: Dataset split (default is TRAIN).
            n_evolutions: Number of evolutions to apply.
            update_split: Whether to update the dataset split.
            batch_size: Batch size for async generation.

        Returns:
            List of generated samples.
        """
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
