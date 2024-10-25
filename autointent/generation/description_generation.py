import asyncio
import random
from collections import defaultdict

from openai import AsyncOpenAI

from autointent.context.data_handler.schemas import Dataset, Intent, Utterance, UtteranceType
from autointent.generation.prompt_scheme import PromptDescription


def get_utterances_by_id(utterances: list[Utterance]) -> dict[int, list[str]]:
    """
    Groups utterances by their labels.

    Args:
        utterances (list[Utterance]): List of utterances with `label` and `text` attributes.

    Returns:
        dict[int, list[str]]: A dictionary where labels map to lists of utterance texts.
    """
    intent_utterances = defaultdict(list)

    for utterance in utterances:
        if utterance.type == UtteranceType.oos:
            continue

        text = utterance.text
        if utterance.type == UtteranceType.multilabel:
            for label in utterance.label:
                intent_utterances[label].append(text)
        else:
            intent_utterances[utterance.label].append(text)

    return intent_utterances


async def create_intent_description(
    client: AsyncOpenAI,
    intent_name: str | None,
    utterances: list[str],
    regexp_patterns: list[str],
    prompt: PromptDescription,
    model_name: str,
) -> str:
    """
    Generates a description for a specific intent using an OpenAI model.

    Args:
        client (AsyncOpenAI): The OpenAI client instance used to communicate with the model.
        intent_name (str | None): The name of the intent to describe. If None, an empty string will be used.
        utterances (list[str]): A list of example utterances related to the intent.
        regexp_patterns (list[str]): A list of regular expression patterns associated with the intent.
        prompt (PromptDescription): A string template for the prompt, which must include placeholders for {intent_name}
                                  and {user_utterances} to format the content sent to the model.
        model_name (str): The identifier of the OpenAI model to use for generating the description.

    Returns:
        str: The generated description of the intent.

    Raises:
        ValueError: If the response from the model is not a string or is in an unexpected format.
    """
    intent_name = intent_name if intent_name is not None else ""
    utterances = random.sample(utterances, min(5, len(utterances)))
    regexp_patterns = random.sample(regexp_patterns, min(3, len(regexp_patterns)))

    content = prompt.text.format(
        intent_name=intent_name, user_utterances="\n".join(utterances), regexp_patterns="\n".join(regexp_patterns)
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
    """
    Generates descriptions for a list of intents using an OpenAI model.

    Args:
        client (AsyncOpenAI): The OpenAI client used to generate the descriptions.
        intent_utterances (dict[int, list[str]]): A dictionary mapping intent IDs to their corresponding utterances.
        intents (list[Intent]): A list of intents to generate descriptions for.
        prompt (PromptDescription): A string template for the prompt, which must include placeholders for {intent_name}
                                  and {user_utterances} to format the content sent to the model.
        model_name (str): The name of the OpenAI model to use for generating descriptions.

    Returns:
        list[Intent]: The list of intents with updated descriptions.
    """
    tasks = []
    for intent in intents:
        if intent.description is not None:
            continue
        utterances = intent_utterances.get(intent.id, [])
        regexp_patterns = intent.regexp_full_match + intent.regexp_partial_match
        task = asyncio.create_task(
            create_intent_description(
                client=client,
                intent_name=intent.name,
                utterances=utterances,
                regexp_patterns=regexp_patterns,
                prompt=prompt,
                model_name=model_name,
            )
        )
        tasks.append((intent, task))

    descriptions = await asyncio.gather(*(task for _, task in tasks))
    for (intent, _), description in zip(tasks, descriptions, strict=False):
        intent.description = description
    return intents


def enhance_dataset_with_descriptions(
    dataset: Dataset,
    client: AsyncOpenAI,
    prompt: PromptDescription,
    model_name: str = "gpt-4o-mini",
) -> Dataset:
    """
    Enhances a dataset by generating descriptions for intents using an OpenAI model.

    Args:
        dataset (Dataset): The dataset containing utterances and intents that require descriptions.
        client (AsyncOpenAI): The OpenAI client used to generate the descriptions.
        prompt (PromptDescription): A string template for the prompt, which must include placeholders for {intent_name}
                                  and {user_utterances} to format the content sent to the model.
        model_name (str, optional): The OpenAI model to use for generating descriptions. Defaults to "gpt-3.5-turbo".

    Returns:
        Dataset: The dataset with intents enhanced by generated descriptions.
    """
    intent_utterances = get_utterances_by_id(utterances=dataset.utterances)
    dataset.intents = asyncio.run(
        generate_intent_descriptions(client, intent_utterances, dataset.intents, prompt, model_name)
    )
    return dataset
