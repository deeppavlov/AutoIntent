import asyncio
import random
from collections import defaultdict

from openai import AsyncOpenAI

from autointent.context.data_handler.schemas import Dataset, Intent, Utterance, UtteranceType
from autointent.generation.prompt_scheme import PromptDescription

def test_functions(
        arg1: int,
        arg2: str,
) -> bool:
    """
    Summary line.

    Extended description of function.

    :param arg1: Description of arg1
    :type arg1: int
    :param arg2: Description of arg2
    :type arg2: str
    :returns: Description of return value
    :rtype: bool
    """
    pass

def get_utterances_by_id(utterances: list[Utterance]) -> dict[int, list[str]]:
    """
    Groups utterances by their labels.

    :param utterances: List of utterances with `label` and `text` attributes.
    :type utterances: list[Utterance]

    :returns: A dictionary where labels map to lists of utterance texts.
    :rtype: dict[int, list[str]]
    """
    intent_utterances = defaultdict(list)

    for utterance in utterances:
        if utterance.type == UtteranceType.oos:
            continue

        text = utterance.text
        if utterance.type == UtteranceType.multilabel:
            for label in utterance.label:  # type: ignore[union-attr]
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

    :param client: The OpenAI client instance used to communicate with the model.
    :type client: AsyncOpenAI
    :param intent_name: The name of the intent to describe. If None, an empty string will be used.
    :type intent_name: str | None
    :param utterances: A list of example utterances related to the intent.
    :type utterances: list[str]
    :param regexp_patterns: A list of regular expression patterns associated with the intent.

    :param prompt: A string template for the prompt, which must include placeholders for {intent_name}
                                    and {user_utterances} to format the content sent to the model.
    :type prompt: PromptDescription
    :param model_name: The identifier of the OpenAI model to use for generating the description.
    :type model_name: str

    :returns: The generated description of the intent.
    :rtype: str
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

    :param client: The OpenAI client used to generate the descriptions.
    :type client: AsyncOpenAI
    :param intent_utterances: A dictionary mapping intent IDs to their corresponding utterances.
    :type intent_utterances: dict[int, list[str]]
    :param intents: A list of intents to generate descriptions for.
    :type intents: list[Intent]
    :param prompt: A string template for the prompt, which must include placeholders for {intent_name}
                                      and {user_utterances} to format the content sent to the model.
    :type prompt: PromptDescription
    :param model_name: The name of the OpenAI model to use for generating descriptions.
    :type model_name: str

    :returns: The list of intents with updated descriptions.
    :rtype: list[Intent]
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


    :param dataset: The dataset containing utterances and intents that require descriptions.
    :type dataset: Dataset
    :param client: The OpenAI client used to generate the descriptions.
    :type client: AsyncOpenAI
    :param prompt: A string template for the prompt, which must include placeholders for {intent_name}
                                      and {user_utterances} to format the content sent to the model.
    :type prompt: PromptDescription
    :param model_name: The OpenAI model to use for generating descriptions.
    :type model_name: str, optional

    :returns: The dataset with intents enhanced by generated descriptions.
    :rtype: Dataset
    """
    intent_utterances = get_utterances_by_id(utterances=dataset.utterances)
    dataset.intents = asyncio.run(
        generate_intent_descriptions(client, intent_utterances, dataset.intents, prompt, model_name)
    )
    return dataset
