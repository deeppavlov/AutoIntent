from argparse import ArgumentParser
from typing import Any, Literal
import json
import random

import os
import yaml
from ..generator import Generator
from ..utils import load_prompt, safe_format


def read_json_dataset(file_path: os.PathLike):
    with open(file_path, 'r') as file:
        return json.load(file)


def save_json_dataset(file_path: os.PathLike, intents: list[dict[str, Any]]):
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(file_path, 'w') as file:
        json.dump(intents, file, indent=4, ensure_ascii=False)


EvolutionType = Literal["reasoning", "concretizing", "abstract", "formal", "informal", "funny", "goofy"]


class UtteranceEvolver:
    def __init__(
            self,
            generator: Generator,
            evolutions: list[EvolutionType] = EvolutionType.__args__,
            seed: int = 0
        ):
        self.generator = generator
        self.evolutions = evolutions
        self.prompts = load_prompt("evolutions.yaml")
        random.seed(seed)

    def _evolve(self, utterance: str, intent_name: str, evolution: EvolutionType) -> list[str]:
        messages_yaml = safe_format(
            self.prompts[evolution],
            base_instruction=self.prompts["base_instruction"],
            utterance=utterance,
            intent_name=intent_name
        )
        messages = yaml.safe_load(messages_yaml)
        evolved_utterance = self.generator.get_chat_completion(messages)
        return evolved_utterance

    def __call__(self, utterance: str, intent_name: str, n_evolutions: int = 1) -> list[str]:
        res = []
        for _ in range(n_evolutions):
            evolution = random.choice(self.evolutions)
            res.append(self._evolve(utterance, intent_name, evolution))
        return res


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True, help="Path to json with intent records")
    parser.add_argument('--output-path', type=str, required=True, help="Where to save result")
    parser.add_argument('--n-evolutions', type=int, default=1, help="Number of utterances to generate for each intent")
    parser.add_argument('--reasoning', action="store_true", help="Whether to use `Reasoning` evolution")
    parser.add_argument('--concretizing', action="store_true", help="Whether to use `Concretizing` evolution")
    parser.add_argument('--abstract', action="store_true", help="Whether to use `Abstract` evolution")
    parser.add_argument('--formal', action="store_true", help="Whether to use `Formal` evolution")
    parser.add_argument('--informal', action="store_true", help="Whether to use `Informal` evolution")
    parser.add_argument('--funny', action="store_true", help="Whether to use `Funny` evolution")
    parser.add_argument('--goofy', action="store_true", help="Whether to use `Goofy` evolution")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args() 

    evolutions = []
    if args.reasoning:
        evolutions.append("reasoning")
    if args.concretizing:
        evolutions.append("concretizing")
    if args.abstract:
        evolutions.append("abstract")
    if args.formal:
        evolutions.append("formal")
    if args.informal:
        evolutions.append("informal")
    if args.funny:
        evolutions.append("funny")
    if args.goofy:
        evolutions.append("goofy")

    intents = read_json_dataset(args.input_path)

    generator = UtteranceEvolver(Generator(), evolutions, args.seed)
    for intent_record in intents:
        cur_res = []
        for utterance in intent_record["sample_utterances"]:
            cur_res.extend(generator(utterance, intent_record["intent_name"], args.n_evolutions))
        intent_record["sample_utterances"].extend(cur_res)

    save_json_dataset(args.output_path, intents)


if __name__ == '__main__':
    main()
