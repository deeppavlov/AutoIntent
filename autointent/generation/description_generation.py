import asyncio
from collections import defaultdict

from openai import AsyncOpenAI

from autointent.context.data_handler.schemas import Dataset, Intent, Utterance
from autointent.generation.prompts import PROMPT_DESCRIPTION


def get_utternaces_by_id(utterances: list[Utterance]) -> dict[int, list[str]]:
    """
    Groups utterances by their labels.

    Args:
        utterances (list[Utterance]): List of utterances with `label` and `text` attributes.

    Returns:
        dict[int, list[str]]: A dictionary where labels map to lists of utterance texts.
    """
    intent_utterances = defaultdict(list)

    for utterance in utterances:
        if utterance.label is None:
            continue

        text = utterance.text
        if isinstance(utterance.label, list):
            for label in utterance.label:
                intent_utterances[label].append(text)
        else:
            intent_utterances[utterance.label].append(text)

    return intent_utterances


def check_prompt_description(prompt: str) -> None:
    if prompt.find("{intent_name}") == -1 or prompt.find("{user_utterances}") == -1:
        error_text = (
            "The 'prompt_description' template must properly include {intent_name} and {user_utterances} placeholders."
        )
        raise ValueError(error_text)


async def create_intent_description(
    client: AsyncOpenAI, intent_name: str | None, utterances: list[str], prompt: str, model_name: str
) -> str:
    """
    Generates a description for a specific intent using an OpenAI model.

    Args:
        client (AsyncOpenAI): The OpenAI client instance used to communicate with the model.
        intent_name (str | None): The name of the intent to describe. If None, an empty string will be used.
        utterances (list[str]): A list of example utterances related to the intent.
        prompt (str): A string template for the prompt, which must include placeholders for {intent_name}
                                  and {user_utterances} to format the content sent to the model.
        model_name (str): The identifier of the OpenAI model to use for generating the description.

    Returns:
        str: The generated description of the intent.

    Raises:
        ValueError: If the response from the model is not a string or is in an unexpected format.
    """
    intent_name = intent_name if intent_name is not None else ""
    content = prompt.format(intent_name=intent_name, user_utterances="\n".join(utterances[:5]))
    chat_completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": content}],
        model=model_name,
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
    prompt: str,
    model_name: str,
) -> list[Intent]:
    """
    Generates descriptions for a list of intents using an OpenAI model.

    Args:
        client (AsyncOpenAI): The OpenAI client used to generate the descriptions.
        intent_utterances (dict[int, list[str]]): A dictionary mapping intent IDs to their corresponding utterances.
        intents (list[Intent]): A list of intents to generate descriptions for.
        prompt (str): A string template for the prompt, which must include placeholders for {intent_name}
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
        task = asyncio.create_task(
            create_intent_description(
                client=client,
                intent_name=intent.name,
                utterances=utterances,
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
    api_base: str,
    api_key: str,
    prompt: str = PROMPT_DESCRIPTION,
    model_name: str = "gpt-4o-mini",
) -> Dataset:
    """
    Enhances a dataset by generating descriptions for intents using an OpenAI model.

    Args:
        dataset (Dataset): The dataset containing utterances and intents that require descriptions.
        api_base (str): The base URL for the OpenAI API.
        api_key (str): The API key for authenticating the OpenAI client.
        prompt (str): A string template for the prompt, which must include placeholders for {intent_name}
                                  and {user_utterances} to format the content sent to the model.
        model_name (str, optional): The OpenAI model to use for generating descriptions. Defaults to "gpt-3.5-turbo".

    Returns:
        Dataset: The dataset with intents enhanced by generated descriptions.
    """
    check_prompt_description(prompt)

    client = AsyncOpenAI(
        base_url=api_base,
        api_key=api_key,
    )
    intent_utterances = get_utternaces_by_id(utterances=dataset.utterances)
    dataset.intents = asyncio.run(
        generate_intent_descriptions(client, intent_utterances, dataset.intents, prompt, model_name)
    )
    return dataset
