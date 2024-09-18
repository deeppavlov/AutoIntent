from argparse import ArgumentParser
from typing import Any, Literal
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


LengthType = Literal["none", "same", "longer", "shorter"]
StyleType = Literal["none", "formal", "informal", "playful"]


class UtteranceGenerator:
    def __init__(
            self,
            generator: Generator,
            custom_instruction: list[str],
            length: LengthType,
            style: StyleType,
            same_punctuation: bool
        ):
        self.generator = generator
        prompt_template_yaml = load_prompt("generate_utterances.yaml")
        self.prompt_template_yaml = add_extra_instructions(prompt_template_yaml, custom_instruction, length, style, same_punctuation)

    def _generate(self, intent_name: str, example_utterances: list[str], n_examples: int) -> list[str]:
        messages_yaml = safe_format(
            self.prompt_template_yaml,
            intent_name=intent_name, 
            example_utterances=format_utterances(example_utterances),
            n_examples=n_examples
        )
        messages = yaml.safe_load(messages_yaml)
        response_text = self.generator.get_chat_completion(messages)
        return extract_utterances(response_text)
        
    def __call__(self, intent_record: dict[str, Any], n_examples: int, inplace: bool = True) -> list[str]:
        intent_name = intent_record.get("intent_name", "")
        example_utterances = intent_record.get("sample_utterances", [])
        res_utterances = self._generate(intent_name, example_utterances, n_examples)
        if inplace:
            intent_record["sample_utterances"] = intent_record.get("sample_utterances", []) + res_utterances
        return res_utterances

    
def add_extra_instructions(
        prompt_template_yaml: str,
        custom_instruction: list[str],
        length: LengthType,
        style: StyleType,
        same_punctuation: bool
    ) -> str:
    instructions = json.loads(load_prompt("extra_instructions.json"))

    extra_instructions = []
    if length != "none":
        extra_instructions.append(instructions["length"][length])
    if style != "none":
        extra_instructions.append(instructions["style"][style])
    if same_punctuation:
        extra_instructions.append(instructions["punctuation"])

    extra_instructions.extend(custom_instruction)

    parsed_extra_instructions = "\n    ".join([f"- {s}" for s in extra_instructions])
    formatted_prompt = safe_format(prompt_template_yaml, extra_instructions=parsed_extra_instructions)
    return formatted_prompt


def format_utterances(utterances: list[str]) -> str:
    """
    Return
    ---
    str of the following format:

    ```
        1. I want to order a large pepperoni pizza.
        2. Can I get a medium cheese pizza with extra olives?
        3. Please deliver a small veggie pizza to my address.
    ```

    Note
    ---
    tab is inserted before each line because of how yaml processes multi-line fields
    """
    return "\n    ".join(f"{i}. {ut}" for i, ut in enumerate(utterances))


def extract_utterances(response_text: str) -> list[str]:
    """
    Input
    ---
    str of the following format:

    ```
    1. I want to order a large pepperoni pizza.
    2. Can I get a medium cheese pizza with extra olives?
    3. Please deliver a small veggie pizza to my address.
    ```
    
    """
    raw_utterances = response_text.split("\n")
    # remove enumeration
    utterances = [ut[ut.find(" ")+1:] for ut in raw_utterances]
    return utterances


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True, help="Path to json with intent records")
    parser.add_argument('--output-path', type=str, required=True, help="Where to save result")
    parser.add_argument('--n-shots', type=int, required=True, help="Number of utterances to generate for each intent")
    parser.add_argument('--custom-instruction', type=str, action="append", help="Add extra instructions to default prompt. You can use this argument multiple times to add multiple instructions")
    parser.add_argument('--length', choices=["none", "same", "longer", "shorter"], default="none", help="How to extend the prompt with length instruction")
    parser.add_argument('--style', choices=["none", "formal", "informal", "playful"], default="none", help="How to extend the prompt with style instruction")
    parser.add_argument('--same-punctuation', action="store_true", help="Whether to extend the prompt with punctuation instruction")
    args = parser.parse_args()

    intents = read_json_dataset(args.input_path)

    generator = UtteranceGenerator(Generator(), args.custom_instruction, args.length, args.style, args.same_punctuation)
    for intent_record in intents:
        generator(intent_record, args.n_shots, inplace=True)

    save_json_dataset(args.output_path, intents)


if __name__ == '__main__':
    main()
