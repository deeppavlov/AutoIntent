"""Module for generating intent descriptions using OpenAI models.

This module provides functionality to generate descriptions for intents using OpenAI's
language models. It includes utilities for grouping utterances, creating descriptions
for individual intents, and enhancing datasets with generated descriptions.
"""

import asyncio
import random
from collections import defaultdict

from autointent import Dataset
from autointent.generation import Generator
from autointent.generation.chat_templates import PromptDescription
from autointent.schemas import Intent, Sample


def group_utterances_by_label(samples: list[Sample]) -> dict[int, list[str]]:
    """Group utterances from samples by their corresponding labels.

    Args:
        samples: List of samples, each containing a label and utterance.

    Returns:
        Dictionary mapping label IDs to lists of utterances.
    """
    label_mapping = defaultdict(list)

    for sample in samples:
        match sample.label:
            case list():
                # Handle one-hot encoding
                for class_id, label in enumerate(sample.label):
                    if label:
                        label_mapping[class_id].append(sample.utterance)
            case int():
                label_mapping[sample.label].append(sample.utterance)

    return label_mapping


async def create_intent_description(
    client: Generator,
    intent_name: str | None,
    utterances: list[str],
    prompt: PromptDescription,
) -> str:
    """Generate a description for a specific intent using an OpenAI model.

    Args:
        client: OpenAI client instance for model communication.
        intent_name: Name of the intent to describe (empty string if None).
        utterances: Example utterances related to the intent.
        prompt: Template for model prompt with placeholders for intent_name,
               user_utterances, and regex_patterns.

    Raises:
        TypeError: If the model response is not a string.
    """
    intent_name = intent_name if intent_name is not None else ""
    utterances = random.sample(utterances, min(5, len(utterances)))

    return await client.get_chat_completion_async(
        messages=prompt.to_messages(intent_name, utterances),
    )


async def generate_intent_descriptions(
    client: Generator,
    intent_utterances: dict[int, list[str]],
    intents: list[Intent],
    prompt: PromptDescription,
) -> list[Intent]:
    """Generate descriptions for multiple intents using an OpenAI model.

    Args:
        client: OpenAI client for generating descriptions.
        intent_utterances: Dictionary mapping intent IDs to utterances.
        intents: List of intents needing descriptions.
        prompt: Template for model prompt with placeholders for intent_name,
               user_utterances, and regex_patterns.
    """
    tasks = []
    for intent in intents:
        if intent.description is not None:
            continue
        utterances = intent_utterances.get(intent.id, [])
        task = asyncio.create_task(
            create_intent_description(
                client=client,
                intent_name=intent.name,
                utterances=utterances,
                prompt=prompt,
            ),
        )
        tasks.append((intent, task))

    descriptions = await asyncio.gather(*(task for _, task in tasks))
    for (intent, _), description in zip(tasks, descriptions, strict=False):
        intent.description = description
    return intents


def generate_descriptions(
    dataset: Dataset,
    client: Generator,
    prompt: PromptDescription | None = None,
) -> Dataset:
    """Add LLM-generated text descriptions to dataset's intents.

    Args:
        dataset: Dataset containing utterances and intents needing descriptions.
        client: OpenAI client for generating descriptions.
        prompt: Template for model prompt with placeholders for intent_name,
               user_utterances, and regex_patterns.

    See :ref:`intent_description_generation` tutorial.
    """
    samples = []
    for split in dataset.values():
        samples.extend([Sample(**sample) for sample in split.to_list()])
    intent_utterances = group_utterances_by_label(samples)
    if prompt is None:
        prompt = PromptDescription()
    dataset.intents = asyncio.run(
        generate_intent_descriptions(client, intent_utterances, dataset.intents, prompt),
    )
    return dataset
