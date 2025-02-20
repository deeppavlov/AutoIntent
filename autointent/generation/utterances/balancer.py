"""Module for balancing datasets through augmentation of underrepresented classes."""

from collections import defaultdict
from typing import List

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation.utterances.evolution.evolver import UtteranceEvolver
from autointent.generation.utterances.generator import Generator
from autointent.generation.utterances.basic.utterance_generator import UtteranceGenerator


class DatasetBalancer:
    """Class for balancing dataset through example augmentation."""

class DatasetBalancer:
    def __init__(
        self,
        generator: Generator,
        evolutions: List,
        seed: int = 42,
        async_mode: bool = False,
        max_samples_per_class: int | None = None,
    ) -> None:
        if not isinstance(generator, Generator):
            raise TypeError("Generator must be an instance of autointent.generation.utterances.generator.Generator")
        
        if not isinstance(evolutions, list) or not all(callable(e) for e in evolutions):
            raise TypeError("Evolutions must be a list of callable objects")
        
        if max_samples_per_class is not None and max_samples_per_class <= 0:
            raise ValueError("max_samples_per_class must be a positive integer or None")
        
        self.evolver = UtteranceGenerator(generator, evolutions, async_mode)
        self.max_samples = max_samples_per_class


    def balance(
        self, dataset: Dataset, split: str = Split.TRAIN, n_evolutions: int = 3, batch_size: int = 4
    ) -> Dataset:
        """
        Balances the specified dataset split.

        :param dataset: Source dataset
        :param split: Target split for balancing
        :param n_evolutions: Number of augmentations per example
        :param batch_size: Batch size for asynchronous processing
        :return: Balanced dataset
        """
        if dataset.multilabel:
            msg = "Method supports only single-label datasets"
            raise ValueError(msg)

        class_counts = self._count_class_examples(dataset, split)
        max_count = max(class_counts.values())
        target_count = self.max_samples if self.max_samples is not None else max_count
        print(f"Target count per class: {target_count}")  # Добавить логирование

        for class_id, current_count in class_counts.items():
            if current_count < target_count:
                needed = target_count - current_count
                self._augment_class(dataset, split, class_id, needed, n_evolutions, batch_size)

        return dataset

    def _count_class_examples(self, dataset: Dataset, split: str) -> dict[int, int]:
        """Count the number of examples for each class."""
        counts = defaultdict(int)
        for sample in dataset[split]:
            counts[sample[Dataset.label_feature]] += 1
        return counts

    def _augment_class(
        self, dataset: Dataset, split: str, class_id: int, needed: int, n_evolutions: int, batch_size: int
    ) -> None:
        """Generate additional examples for the class."""
        print("\n📂 DATASET BEFORE AUGMENTATION:")
        self._print_dataset(dataset, split)
        intent = next(i for i in dataset.intents if i.id == class_id)
        class_name = getattr(intent, 'name', f'class_{class_id}')  # Получаем имя класса, если доступно
        print(f"\n🚀 Starting augmentation for class {class_id} ({class_name})")
        print(f"📊 Initial samples: {len([s for s in dataset[split] if s[Dataset.label_feature] == class_id])}")
        print(f"🎯 Target needed: {needed} samples")

        class_samples = [s for s in dataset[split] if s[Dataset.label_feature] == class_id]
        if not class_samples:
            msg = f"No samples for class {class_id}"
            raise ValueError(msg)

        per_sample_evolutions = max(1, needed // len(class_samples))
        total_generated = 0

        while total_generated < needed:
            print(f"\n🔄 Batch generation: {per_sample_evolutions} evolutions per sample")
            
            generated = self.evolver.augment(
                dataset, split_name=split, n_generations=per_sample_evolutions, update_split=True, batch_size=batch_size
            )
            print("\n📦 DATASET AFTER EVOLVATION:")
            self._print_dataset(dataset, split)
            print(f"✅ Generated {len(generated)} examples")
            if generated:
                print("🔠 Example generated utterances:")
                for i, example in enumerate(generated[:3]): 
                    utterance = getattr(example, Dataset.utterance_feature, str(example))
                    print(f"   {i+1}. {utterance[:60]}...") 
                    
            total_generated += len(generated)
            print(f"📈 Progress: {total_generated}/{needed} ({min(100, int(total_generated/needed*100))}%)")

            if total_generated > needed:
                removed = total_generated - needed
                self._remove_extra_samples(dataset, split, class_id, removed)
                print(f"✂️ Removed {removed} extra examples to match target")

        final_count = len([s for s in dataset[split] if s[Dataset.label_feature] == class_id])
        print(f"\n🎉 Completed augmentation for class {class_id} ({class_name})")
        print(f"📦 Total samples after augmentation: {final_count}")
        print("\n📦 DATASET AFTER AUGMENTATION:")
        self._print_dataset(dataset, split)
        print("━" * 50)
        

    def _remove_extra_samples(self, dataset: Dataset, split: str, class_id: int, extra: int) -> None:
        """Remove extra examples of the class."""
        class_indices = [i for i, s in enumerate(dataset[split]) if s[Dataset.label_feature] == class_id]
        indices_to_remove = class_indices[-extra:]

        new_data = [s for i, s in enumerate(dataset[split]) if i not in indices_to_remove]
        dataset[split] = dataset[split].from_list(new_data)
    def _print_dataset(self, dataset: Dataset, split: str) -> None:
            """Helper method to print dataset in readable format"""
            print(f"Split: {split}")
            print("-" * 50)
            for i, sample in enumerate(dataset[split]):
                label = sample[Dataset.label_feature]
                text = sample[Dataset.utterance_feature]
                print(f"{i+1:3d} | {label:15} | {text[:50]:<50}...")
            print("-" * 50)
            print(f"Total samples: {len(dataset[split])}\n")