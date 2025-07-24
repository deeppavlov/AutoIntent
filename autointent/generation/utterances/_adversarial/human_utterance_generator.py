from collections import defaultdict

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation import Generator
from autointent.generation.chat_templates._evolution_templates_schemas import Message, Role
from autointent.schemas import Sample

from .critic_human_like import CriticHumanLike


class HumanUtteranceGenerator:
    """Generator of human-like utterances.

    This class rewrites given user utterances to make them sound more natural and human-like,
    while preserving their original intent. The generation process is iterative and attempts
    to bypass a critic that identifies machine-generated text.
    """

    def __init__(self, generator: Generator, critic: CriticHumanLike)-> None:
        """Initialize the CritlUtteranceGenerator.

        Args:
            generator: Wrapper for the LLM API used to generate utterances.
            critic: Critic to determine whether the generated utterance sounds human-like.
        """
        self.generator = generator
        self.critic = critic

    def augment(
            self,
            dataset: Dataset,
            split_name: str = Split.TRAIN,
            update_split: bool = True,
            n_final_per_class: int = 5
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
        original_split = dataset[split_name]
        id_to_name = {intent.id: intent.name for intent in dataset.intents}
        new_samples = []

        class_to_samples = defaultdict(list)
        for sample in original_split:
            class_to_samples[sample["label"]].append(sample["utterance"])

        for intent_id, intent_name in id_to_name.items():
            generated_count = 0
            attempt = 0

            seed_utterances = class_to_samples.get(intent_id, [])
            if not seed_utterances:
                continue

            while generated_count < n_final_per_class and attempt < n_final_per_class * 3:
                attempt += 1
                seed = seed_utterances[attempt % len(seed_utterances)]
                rejected = []

                for _ in range(3):
                    prompt = self._build_adversarial_prompt(seed, intent_name, rejected)
                    generated = self.generator.get_chat_completion([prompt]).strip()

                    if self.critic.is_human(generated, intent_name):
                        new_samples.append({
                            Dataset.label_feature: intent_id,
                            Dataset.utterance_feature: generated
                        })
                        generated_count += 1
                        break
                    rejected.append(generated)

        if update_split:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])

        return [Sample(**sample) for sample in new_samples]

    def _build_adversarial_prompt(self, example: str, intent_name: str, rejected: list[str]) -> Message:
        """Build an adversarial prompt to guide the model in generating more human-like utterances.

        Args:
            example: The original utterance to be modified.
            intent_name: The intent of the utterance.
            rejected: List of previously rejected generations.

        Returns:
            Message: A formatted prompt guiding the generator to improve naturalness.
        """
        rejected_block = "\n".join(f"- {r}" for r in rejected) if rejected else "None"
        content = (
            "Your task is to rewrite the following user utterance so that it sounds as natural "
            "and human-like as possible, while preserving its original intent: "
            f"'{intent_name}'.\n\n"
            f'Original utterance: "{example}"\n\n'
            f"The following previous attempts were classified as machine-generated and rejected:\n{rejected_block}\n\n"
            "Try to write something that would pass as written by a real human. Output a single version only.\n"
            "IMPORTANT: You must modify the original utterance."
        )
        return Message(role=Role.USER, content=content)
