"""Module for balancing datasets through augmentation of underrepresented classes."""

import logging
from collections import defaultdict

from datasets import Dataset as HFDataset

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation import Generator
from autointent.generation.chat_templates import BaseSynthesizerTemplate

from .utterance_generator import UtteranceGenerator

logger = logging.getLogger(__name__)


class DatasetBalancer:
    """Balance dataset's classes distribution.

    If your dataset is unbalanced, you can add LLM-generated samples.
    This method uses :py:class:`autointent.generation.utterances.UtteranceGenerator` under the hood.

    See tutorial :ref:`balancer_aug` for usage examples.

    Args:
        generator (Generator): The generator object used to create utterances.
        prompt_maker (Callable[[Intent, int], list[Message]]): A callable that creates prompts for the generator.
        async_mode (bool, optional): Whether to run the generator in asynchronous mode. Defaults to False.
        max_samples_per_class (int | None, optional): The maximum number of samples per class.
            Must be a positive integer or None. Defaults to None.
    """

    def __init__(
        self,
        generator: Generator,
        prompt_maker: BaseSynthesizerTemplate,
        async_mode: bool = False,
        max_samples_per_class: int | None = None,
    ) -> None:
        if max_samples_per_class is not None and max_samples_per_class <= 0:
            msg = "max_samples_per_class must be a positive integer or None"
            raise ValueError(msg)

        self.utterance_generator = UtteranceGenerator(
            generator=generator, prompt_maker=prompt_maker, async_mode=async_mode
        )
        self.max_samples = max_samples_per_class

    def balance(self, dataset: Dataset, split: str = Split.TRAIN, batch_size: int = 4) -> Dataset:
        """Balances the specified dataset split.

        Args:
            dataset: Source dataset
            split: Target split for balancing
            batch_size: Batch size for asynchronous processing
        """
        if dataset.multilabel:
            msg = "Method supports only single-label datasets"
            raise ValueError(msg)

        class_counts = self._count_class_examples(dataset, split)
        max_count = max(class_counts.values())
        target_count = self.max_samples if self.max_samples is not None else max_count
        logger.debug("Target count per class: %s", target_count)
        for class_id, current_count in class_counts.items():
            if current_count < target_count:
                needed = target_count - current_count
                self._augment_class(dataset, split, class_id, needed, batch_size)

        return dataset

    def _count_class_examples(self, dataset: Dataset, split: str) -> dict[int, int]:
        """Count the number of examples for each class."""
        counts: dict[int, int] = defaultdict(int)
        for sample in dataset[split]:
            counts[sample[Dataset.label_feature]] += 1
        return counts

    def _augment_class(self, dataset: Dataset, split: str, class_id: int, needed: int, batch_size: int) -> None:
        """Generate additional examples for the class."""
        intent = next(i for i in dataset.intents if i.id == class_id)
        class_name = getattr(intent, "name", f"class_{class_id}")
        logger.debug("Starting augmentation for class %s (%s)", class_id, class_name)
        logger.debug("Initial samples: %s", len([s for s in dataset[split] if s[Dataset.label_feature] == class_id]))
        logger.debug("Target needed: %s samples", needed)

        class_samples = [s for s in dataset[split] if s[Dataset.label_feature] == class_id]
        if not class_samples:
            msg = f"No samples for class {class_id}"
            raise ValueError(msg)

        generated_utterances: list[str] = []
        max_attempts = 5
        attempts = 0

        while len(generated_utterances) < needed and attempts < max_attempts:
            current_needed = needed - len(generated_utterances)
            current_batch = min(batch_size, current_needed)
            logger.debug("Attempt %s: Generating %s utterances for class %s", attempts + 1, current_batch, class_id)

            new_utterances = self.utterance_generator(intent_data=intent, n_generations=current_batch)

            valid_utterances = self._process_utterances(new_utterances)
            for ut in valid_utterances:
                if ut and isinstance(ut, str):
                    generated_utterances.append(ut)
                    if len(generated_utterances) >= needed:
                        break

            logger.debug("Generated %s valid utterances in this attempt", len(valid_utterances))
            logger.debug(
                "Progress: %s/%s (%s%%)",
                len(generated_utterances),
                needed,
                min(100, int(len(generated_utterances) / needed * 100)),
            )

            attempts += 1

        if len(generated_utterances) < needed:
            logger.debug(
                "Warning: Could only generate %s/%s utterances after %s attempts",
                len(generated_utterances),
                needed,
                max_attempts,
            )

        generated_utterances = generated_utterances[:needed]

        new_samples = []
        for utterance in generated_utterances:
            new_sample = {Dataset.utterance_feature: utterance, Dataset.label_feature: class_id}
            new_samples.append(new_sample)

        updated_data = list(dataset[split]) + new_samples
        dataset[split] = HFDataset.from_list(updated_data)

        final_count = len([s for s in dataset[split] if s[Dataset.label_feature] == class_id])
        logger.debug("Completed augmentation for class %s (%s)", class_id, class_name)
        logger.debug("Total samples after augmentation: %s", final_count)

    def _process_utterances(self, generated: list[str]) -> list[str]:
        """Process and clean generated utterances.

        Args:
            generated: Generated list
        """
        processed = []
        for ut in generated:
            if "', '" in ut or "',\n" in ut:
                clean_ut = ut.replace("[", "").replace("]", "").replace("'", "")
                split_ut = [u.strip() for u in clean_ut.split(", ") if u.strip()]
                processed.extend(split_ut)
            else:
                processed.append(ut.strip())
        return processed
