import importlib.resources as ires
import json
from typing import Any, Literal

import yaml

from ..generator import Generator
from ..utils import safe_format

LengthType = Literal["none", "same", "longer", "shorter"]
StyleType = Literal["none", "formal", "informal", "playful"]


class UtteranceGenerator:
    def __init__(
        self,
        generator: Generator,
        custom_instruction: list[str],
        length: LengthType,
        style: StyleType,
        same_punctuation: bool,
    ):
        self.generator = generator
        prompt_template_yaml = load_prompt()
        self.prompt_template_yaml = add_extra_instructions(
            prompt_template_yaml,
            custom_instruction,
            length,
            style,
            same_punctuation,
        )

    def _generate(self, intent_name: str, example_utterances: list[str], n_examples: int) -> list[str]:
        messages_yaml = safe_format(
            self.prompt_template_yaml,
            intent_name=intent_name,
            example_utterances=format_utterances(example_utterances),
            n_examples=n_examples,
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


def load_prompt():
    with ires.files("autointent.generation.basic").joinpath("chat_template.yaml").open() as file:
        return file.read()


def load_extra_instructions():
    with ires.files("autointent.generation.basic").joinpath("extra_instructions.json").open() as file:
        return json.load(file)


def add_extra_instructions(
    prompt_template_yaml: str,
    custom_instruction: list[str],
    length: LengthType,
    style: StyleType,
    same_punctuation: bool,
) -> str:
    instructions = load_extra_instructions()

    extra_instructions = []
    if length != "none":
        extra_instructions.append(instructions["length"][length])
    if style != "none":
        extra_instructions.append(instructions["style"][style])
    if same_punctuation:
        extra_instructions.append(instructions["punctuation"])

    extra_instructions.extend(custom_instruction)

    parsed_extra_instructions = "\n    ".join([f"- {s}" for s in extra_instructions])
    return safe_format(prompt_template_yaml, extra_instructions=parsed_extra_instructions)


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
    return [ut[ut.find(" ") + 1 :] for ut in raw_utterances]
