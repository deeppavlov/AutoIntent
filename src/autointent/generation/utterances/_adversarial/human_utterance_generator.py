import asyncio
import logging
import random
from collections import defaultdict
from functools import partial

import aiometer
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation import Generator
from autointent.generation.chat_templates._evolution_templates_schemas import Message, Role
from autointent.schemas import Sample

from .critic_human_like import CriticHumanLike

logger = logging.getLogger(__name__)


class HumanUtteranceGenerator:
    """Generator of human-like utterances.

    This class rewrites given user utterances to make them sound more natural and human-like,
    while preserving their original intent. The generation process is iterative and attempts
    to bypass a critic that identifies machine-generated text.
    """

    def __init__(
        self,
        generator: Generator,
        critic: CriticHumanLike,
        async_mode: bool = False,
        max_at_once: int = 5,
        max_per_second: int = 10,
    ) -> None:
        """Initialize the HumanUtteranceGeneratoror.

        Args:
            generator: Wrapper for the LLM API used to generate utterances.
            critic: Critic to determine whether the generated utterance sounds human-like.
            async_mode: Whether to use asynchronous mode for generation.
            max_at_once: Maximum number of concurrent async tasks.
            max_per_second: Maximum number of tasks per second.
        """
        self.generator = generator
        self.critic = critic
        self.async_mode = async_mode
        self.max_at_once = max_at_once
        self.max_per_second = max_per_second

    def augment(
        self, dataset: Dataset, split_name: str = Split.TRAIN, update_split: bool = True, n_final_per_class: int = 5
    ) -> list[Sample]:
        """Generate human-like utterances for each intent by iteratively refining machine-generated candidates.

        Args:
            dataset: The dataset to augment.
            split_name: The name of the split to augment (e.g., 'train').
            update_split: Whether to update the dataset split with the new utterances.
            n_final_per_class: Number of successful utterances to generate per intent.

        Returns:
            list[Sample]: List of newly generated samples.
        """
        if self.async_mode:
            return asyncio.run(
                self.augment_async(
                    dataset=dataset,
                    split_name=split_name,
                    update_split=update_split,
                    n_final_per_class=n_final_per_class,
                )
            )
        original_split = dataset[split_name]
        id_to_name = {intent.id: intent.name for intent in dataset.intents}
        new_samples = []

        class_to_samples = defaultdict(list)
        for sample in original_split:
            class_to_samples[sample["label"]].append(sample["utterance"])

        for intent_id, intent_name in id_to_name.items():
            if intent_name is None:
                logger.warning("Intent with id %s has no name! Skipping it...", intent_id)
                continue
            generated_count = 0
            attempt = 0

            seed_utterances = class_to_samples.get(intent_id, [])
            if not seed_utterances:
                continue

            while generated_count < n_final_per_class and attempt < n_final_per_class * 3:
                attempt += 1
                n_seeds = min(3, len(seed_utterances))
                seed_examples = random.sample(seed_utterances, k=n_seeds)
                rejected: list[str] = []

                for _ in range(3):
                    prompt = self._build_adversarial_prompt(intent_name, seed_examples, rejected)
                    generated = self.generator.get_chat_completion([prompt]).strip()
                    if self.critic.is_human(generated, intent_name):
                        new_samples.append({Dataset.label_feature: intent_id, Dataset.utterance_feature: generated})
                        generated_count += 1
                        break
                    rejected.append(generated)
        if update_split:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])

        return [Sample(**sample) for sample in new_samples]

    async def augment_async(
        self, dataset: Dataset, split_name: str = Split.TRAIN, update_split: bool = True, n_final_per_class: int = 5
    ) -> list[Sample]:
        original_split = dataset[split_name]
        id_to_name = {intent.id: intent.name for intent in dataset.intents}
        new_samples = []

        class_to_samples = defaultdict(list)
        for sample in original_split:
            class_to_samples[sample["label"]].append(sample["utterance"])

        async def generate_one(intent_id: str, intent_name: str) -> list[dict[str, str]]:
            generated: list[dict[str, str]] = []
            attempts = 0
            seed_utterances = class_to_samples[intent_id]
            while len(generated) < n_final_per_class and attempts < n_final_per_class * 3:
                attempts += 1
                seed_examples = random.sample(seed_utterances, k=min(3, len(seed_utterances)))
                rejected: list[str] = []

                for _ in range(3):
                    prompt = self._build_adversarial_prompt(intent_name, seed_examples, rejected)
                    utterance = (await self.generator.get_chat_completion_async([prompt])).strip()
                    if await self.critic.is_human_async(utterance, intent_name):
                        generated.append({Dataset.label_feature: intent_id, Dataset.utterance_feature: utterance})
                        break
                    rejected.append(utterance)
            return generated

        tasks = [
            partial(generate_one, str(intent_id), intent_name)
            for intent_id, intent_name in id_to_name.items()
            if class_to_samples.get(intent_id) and intent_name is not None
        ]

        results = await aiometer.run_all(
            tasks,
            max_at_once=self.max_at_once,
            max_per_second=self.max_per_second,
        )

        for result in results:
            new_samples.extend(result)
        for s in new_samples:
            s['label'] = int(s['label'])
        if update_split:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])

        return [Sample(**sample) for sample in new_samples]

    def _build_adversarial_prompt(self, intent_name: str, seed_examples: list[str], rejected: list[str]) -> Message:
        """Build a few-shot prompt.

        Build a few-shot prompt to guide the generator to create a new human-like utterance
        from scratch based on the intent name and example utterances.

        Args:
            intent_name: The intent of the utterance.
            seed_examples: List of 1-3 example utterances for the intent.
            rejected: List of previously rejected generations.

        Returns:
            Message: A formatted prompt instructing the generator to produce a new natural-sounding utterance..
        """
        rejected_block = "\n".join(f"- {r}" for r in rejected) if rejected else "None"
        examples_block = "\n".join(f'- "{ex}"' for ex in seed_examples)
        content = (
            f"Your task is to generate a new user utterance that fits the intent '{intent_name}'.\n\n"
            "Here are some examples of utterances for this intent:\n"
            f"{examples_block}\n\n"
            "Preserving its original intent: "
            f"'{intent_name}'.\n\n"
            f"The following previous attempts were classified as machine-generated and rejected:\n{rejected_block}\n\n"
            "Try to write something that would pass as written by a real human. Output a single version only.\n"
            "IMPORTANT: You must modify the original utterance."
        )
        return Message(role=Role.USER, content=content)
