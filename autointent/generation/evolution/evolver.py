import importlib.resources as ires
import random
from typing import Literal

import yaml

from ..generator import Generator
from ..utils import safe_format

EvolutionType = Literal["reasoning", "concretizing", "abstract", "formal", "informal", "funny", "goofy"]


class UtteranceEvolver:
    def __init__(self, generator: Generator, evolutions: list[EvolutionType] = EvolutionType.__args__, seed: int = 0):
        self.generator = generator
        self.evolutions = evolutions
        self.prompts = load_prompts()
        random.seed(seed)

    def _evolve(self, utterance: str, intent_name: str, evolution: EvolutionType) -> list[str]:
        messages_yaml = safe_format(
            self.prompts[evolution],
            base_instruction=self.prompts["base_instruction"],
            utterance=utterance,
            intent_name=intent_name,
        )
        messages = yaml.safe_load(messages_yaml)
        return self.generator.get_chat_completion(messages)

    def __call__(self, utterance: str, intent_name: str, n_evolutions: int = 1) -> list[str]:
        res = []
        for _ in range(n_evolutions):
            evolution = random.choice(self.evolutions)
            res.append(self._evolve(utterance, intent_name, evolution))
        return res


def load_prompts() -> dict[str, str]:
    files = ires.files("autointent.generation.evolution.chat_templates")

    res = {}
    for file_name in ["reasoning.yaml", "concretizing.yaml", "abstract.yaml", "base_instruction.txt"]:
        with files.joinpath(file_name).open() as file:
            res[file_name.split(".")[0]] = file.read()

    return res
