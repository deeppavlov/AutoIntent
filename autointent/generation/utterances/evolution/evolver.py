"""
Evolutionary strategy to augmenting utterances.

Deeply inspired by DeepEval evolutions.
"""

import importlib.resources as ires
import random
from typing import Literal

import yaml
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation.utterances.generator import Generator
from autointent.generation.utterances.utils import safe_format  # type: ignore[attr-defined]
from autointent.schemas import Sample

EvolutionType = Literal["reasoning", "concretizing", "abstract", "formal", "informal", "funny", "goofy"]


class UtteranceEvolver:
    """
    Evolutionary strategy to augmenting utterances.

    Deeply inspired by DeepEval evolutions. This method takes single utterance and prompts LLM
    to change it in a specific way.
    """

    def __init__(self, generator: Generator, evolutions: list[EvolutionType], seed: int = 0) -> None:
        """Initialize."""
        self.generator = generator
        self.evolutions = evolutions
        self.prompts = _load_prompts()
        random.seed(seed)

    def _evolve(self, utterance: str, intent_name: str, evolution: EvolutionType) -> str:
        """Apply evolutions single time."""
        messages_yaml = safe_format(
            self.prompts[evolution],
            base_instruction=self.prompts["base_instruction"],
            utterance=utterance,
            intent_name=intent_name,
        )
        messages = yaml.safe_load(messages_yaml)
        return self.generator.get_chat_completion(messages)

    def __call__(self, utterance: str, intent_name: str, n_evolutions: int = 1) -> list[str]:
        """Apply evolutions mupltiple times."""
        res = []
        for _ in range(n_evolutions):
            evolution = random.choice(self.evolutions)
            res.append(self._evolve(utterance, intent_name, evolution))
        return res

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
            intent_info = next(intent for intent in dataset.intents if intent.id == label)
            generated_utterances = self(
                utterance=utterance, intent_name=intent_info.name or "", n_evolutions=n_evolutions
            )
            new_samples.extend(
                [{Dataset.label_feature: intent_info.id, Dataset.utterance_feature: ut} for ut in generated_utterances]
            )
        if update_split:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])
        return [Sample(**sample) for sample in new_samples]


def _load_prompts() -> dict[str, str]:
    files = ires.files("autointent.generation.utterances.evolution.chat_templates")

    res = {}
    for file_name in ["reasoning.yaml", "concretizing.yaml", "abstract.yaml", "base_instruction.txt"]:
        with files.joinpath(file_name).open() as file:
            res[file_name.split(".")[0]] = file.read()

    return res
