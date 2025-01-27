"""Basic generation of new utterances from existing ones."""

import importlib.resources as ires
import json
import random
from typing import Any, Literal

import yaml
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation.utterances.generator import Generator
from autointent.generation.utterances.utils import safe_format  # type: ignore[attr-defined]
from autointent.schemas import Sample

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

    def augment(
        self,
        dataset: Dataset,
        split_name: str = Split.TRAIN,
        n_generations: int = 5,
        max_sample_utterances: int = 5,
        update_split: bool = True,
    ) -> list[Sample]:
        """
        Augment some split of dataset.

        Note that for now it supports only single-label datasets.
        """
        original_split = dataset[split_name]
        new_samples = []
        for intent in dataset.intents:
            filtered_split = original_split.filter(lambda sample, id=intent.id: sample[Dataset.label_feature] == id)
            sample_utterances = filtered_split[Dataset.utterance_feature]
            if max_sample_utterances is not None:
                sample_utterances = random.sample(sample_utterances, k=max_sample_utterances)
            generated_utterances = self(
                intent_name=intent.name or "",
                example_utterances=sample_utterances,
                n_generations=n_generations,
            )
            new_samples.extend(
                [{Dataset.label_feature: intent.id, Dataset.utterance_feature: ut} for ut in generated_utterances]
            )
        if update_split:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])
        return [Sample(**sample) for sample in new_samples]


def _load_prompt() -> str:
    with ires.files("autointent.generation.utterances.basic").joinpath("chat_template.yaml").open() as file:
        return file.read()


def _load_extra_instructions() -> dict[str, Any]:
    with ires.files("autointent.generation.utterances.basic").joinpath("extra_instructions.json").open() as file:
        return json.load(file)  # type: ignore[no-any-return]


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
    return safe_format(prompt_template_yaml, extra_instructions=parsed_extra_instructions)  # type: ignore[no-any-return]


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
