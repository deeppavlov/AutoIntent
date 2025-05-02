"""Module for generating intent descriptions using OpenAI models.

This module provides functionality to generate descriptions for intents using OpenAI's
language models. It includes utilities for grouping utterances, creating descriptions
for individual intents, and enhancing datasets with generated descriptions.
"""

import asyncio
import random
from collections import defaultdict

from openai import AsyncOpenAI

from autointent import Dataset
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
    client: AsyncOpenAI,
    intent_name: str | None,
    utterances: list[str],
    regex_patterns: list[str],
    prompt: PromptDescription,
    model_name: str,
) -> str:
    """Generate a description for a specific intent using an OpenAI model.

    Args:
        client: OpenAI client instance for model communication.
        intent_name: Name of the intent to describe (empty string if None).
        utterances: Example utterances related to the intent.
        regex_patterns: Regular expression patterns associated with the intent.
        prompt: Template for model prompt with placeholders for intent_name,
               user_utterances, and regex_patterns.
        model_name: Identifier of the OpenAI model to use.

    Raises:
        TypeError: If the model response is not a string.
    """
    intent_name = intent_name if intent_name is not None else ""
    utterances = random.sample(utterances, min(5, len(utterances)))
    regex_patterns = random.sample(regex_patterns, min(3, len(regex_patterns)))

    content = prompt.text.format(
        intent_name=intent_name,
        user_utterances="\n".join(utterances),
        regex_patterns="\n".join(regex_patterns),
    )
    chat_completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": content}],
        model=model_name,
        temperature=0.2,
    )
    result = chat_completion.choices[0].message.content

    if not isinstance(result, str):
        error_text = f"Unexpected response type: expected str, got {type(result).__name__}"
        raise TypeError(error_text)
    return result


async def generate_intent_descriptions(
    client: AsyncOpenAI,
    intent_utterances: dict[int, list[str]],
    intents: list[Intent],
    prompt: PromptDescription,
    model_name: str,
) -> list[Intent]:
    """Generate descriptions for multiple intents using an OpenAI model.

    Args:
        client: OpenAI client for generating descriptions.
        intent_utterances: Dictionary mapping intent IDs to utterances.
        intents: List of intents needing descriptions.
        prompt: Template for model prompt with placeholders for intent_name,
               user_utterances, and regex_patterns.
        model_name: Name of the OpenAI model to use.
    """
    tasks = []
    for intent in intents:
        if intent.description is not None:
            continue
        utterances = intent_utterances.get(intent.id, [])
        regex_patterns = intent.regex_full_match + intent.regex_partial_match
        task = asyncio.create_task(
            create_intent_description(
                client=client,
                intent_name=intent.name,
                utterances=utterances,
                regex_patterns=regex_patterns,
                prompt=prompt,
                model_name=model_name,
            ),
        )
        tasks.append((intent, task))

    descriptions = await asyncio.gather(*(task for _, task in tasks))
    for (intent, _), description in zip(tasks, descriptions, strict=False):
        intent.description = description
    return intents


def generate_descriptions(
    dataset: Dataset,
    client: AsyncOpenAI,
    model_name: str,
    prompt: PromptDescription | None = None,
) -> Dataset:
    """Add LLM-generated text descriptions to dataset's intents.

    Args:
        dataset: Dataset containing utterances and intents needing descriptions.
        client: OpenAI client for generating descriptions.
        prompt: Template for model prompt with placeholders for intent_name,
               user_utterances, and regex_patterns.
        model_name: OpenAI model identifier for generating descriptions.

    See :ref:`intent_description_generation` tutorial.
    """
    samples = []
    for split in dataset.values():
        samples.extend([Sample(**sample) for sample in split.to_list()])
    intent_utterances = group_utterances_by_label(samples)
    if prompt is None:
        prompt = PromptDescription()
    dataset.intents = asyncio.run(
        generate_intent_descriptions(client, intent_utterances, dataset.intents, prompt, model_name),
    )
    return dataset
