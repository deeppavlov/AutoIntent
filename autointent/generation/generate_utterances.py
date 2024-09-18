from argparse import ArgumentParser
from typing import Any
import json

import os
import yaml
from .generator import Generator
from .utils import load_prompt, safe_format


def read_json_dataset(file_path: os.PathLike):
    with open(file_path, 'r') as file:
        return json.load(file)


def save_json_dataset(file_path: os.PathLike, intents: list[dict[str, Any]]):
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(file_path, 'w') as file:
        json.dump(intents, file, indent=4, ensure_ascii=False)


class UtteranceGenerator:
    def __init__(self, generator: Generator):
        self.generator = generator
        self.prompt_template_yaml = load_prompt("generate_utterances.yaml")


    def _generate(self, intent_name: str, example_utterances: list[str], n_examples: int) -> list[str]:
        messages_yaml = safe_format(
            self.prompt_template_yaml,
            intent_name=intent_name, 
            example_utterances="\n    ".join(example_utterances),
            n_examples=n_examples
        )
        messages = yaml.safe_load(messages_yaml)
        response_text = self.generator.get_chat_completion(messages)
        return response_text.split("\n")
        
    def __call__(self, intent_record: dict[str, Any], n_examples: int, inplace: bool = True) -> list[str]:
        intent_name = intent_record.get("intent_name", "")
        example_utterances = intent_record.get("sample_utterances", [])
        res_utterances = self._generate(intent_name, example_utterances, n_examples)
        if inplace:
            intent_record["sample_utterances"] = intent_record.get("sample_utterances", []) + res_utterances
        return res_utterances


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True, help="Path to json with intent records")
    parser.add_argument('--output-path', type=str, required=True, help="Where to save result")
    parser.add_argument('--n-shots', type=int, required=True, help="Number of utterances to generate for each intent")
    args = parser.parse_args()

    intents = read_json_dataset(args.input_path)

    generator = UtteranceGenerator(Generator())
    for intent_record in intents:
        generator(intent_record, args.n_shots, inplace=True)

    save_json_dataset(args.output_path, intents)


if __name__ == '__main__':
    main()
