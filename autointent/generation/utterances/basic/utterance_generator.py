"""Basic generation of new utterances from existing ones."""

import importlib.resources as ires
import json
from typing import Any, Literal

import yaml

from autointent.generation.utterances.generator import Generator
from autointent.generation.utterances.utils import safe_format

LengthType = Literal["none", "same", "longer", "shorter"]
StyleType = Literal["none", "formal", "informal", "playful"]


class UtteranceGenerator:
    """
    Basic generation of new utterances from existing ones.

    This augmentation method simply prompts LLM to look at existing examples
    and generate similar. Additionaly it can consider some aspects of style,
    punctuation and length of the desired generations.
    """

    def __init__(
        self,
        generator: Generator,
        custom_instruction: list[str],
        length: LengthType,
        style: StyleType,
        same_punctuation: bool,
    ) -> None:
        """Initialize."""
        self.generator = generator
        prompt_template_yaml = _load_prompt()
        self.prompt_template_yaml = _add_extra_instructions(
            prompt_template_yaml,
            custom_instruction,
            length,
            style,
            same_punctuation,
        )

    def __call__(self, intent_name: str, example_utterances: list[str], n_generations: int) -> list[str]:
        """Generate new utterances."""
        messages_yaml = safe_format(
            self.prompt_template_yaml,
            intent_name=intent_name,
            example_utterances=_format_utterances(example_utterances),
            n_examples=n_generations,
        )
        messages = yaml.safe_load(messages_yaml)
        response_text = self.generator.get_chat_completion(messages)
        return _extract_utterances(response_text)


def _load_prompt() -> str:
    with ires.files("autointent.generation.basic").joinpath("chat_template.yaml").open() as file:
        return file.read()


def _load_extra_instructions() -> dict[str, Any]:
    with ires.files("autointent.generation.basic").joinpath("extra_instructions.json").open() as file:
        return json.load(file)


def _add_extra_instructions(
    prompt_template_yaml: str,
    custom_instruction: list[str],
    length: LengthType,
    style: StyleType,
    same_punctuation: bool,
) -> str:
    instructions = _load_extra_instructions()

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


def _format_utterances(utterances: list[str]) -> str:
    """
    Convert given utterances into string that is ready to insert into prompt.

    Given list of utterances, the output string is returned in the following format:
    .. code-block::
        1. I want to order a large pepperoni pizza.
        2. Can I get a medium cheese pizza with extra olives?
        3. Please deliver a small veggie pizza to my address.

    Note that tab is inserted before each line because of how yaml processes multi-line fields.
    """
    return "\n    ".join(f"{i}. {ut}" for i, ut in enumerate(utterances))


def _extract_utterances(response_text: str) -> list[str]:
    """
    Parse LLM output.

    Inverse function to :py:func:`_format_utterances`.
    """
    raw_utterances = response_text.split("\n")
    # remove enumeration
    return [ut[ut.find(" ") + 1 :] for ut in raw_utterances]
