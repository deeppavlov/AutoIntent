import asyncio
from collections import defaultdict

from openai import AsyncOpenAI

from autointent.context.data_handler.schemas import Dataset, Intent, Utterance

PROMPT_DESCRIPTION = """
Your task is to write a description of the intent.

You are given the name of the intent and user intentions related to it. The description should be:
1) in declarative form
2) no more than one sentence
3) in the language in which the utterances and the name are written.

Remember:
Respond with just the description, no extra details.
Keep in mind that either the names or user queries may not be provided.

For example:

name:
activate_my_card
user utterances:
Please help me with my card. It won't activate.
I tried but am unable to activate my card.
I want to start using my card.
description:
user wants to activate his card

name:
beneficiary_not_allowed
user utterances:

description:
user want to know why his beneficiary is not allowed

name:
оформление_отпуска
user utterances:
как оформить отпуск
в какие даты надо оформить отпуск
как запланировать отпуск
description:
пользователь спрашивает про оформление отпуска

name:
{intent_name}
user utterances:
{user_utterances}
description:
"""


def get_utternaces_by_id(utterances: list[Utterance]) -> dict[int, list[str]]:
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


async def generate_intent_description(
    client: AsyncOpenAI, intent_name: str, utterances: list[str], model_name: str
) -> str:
    content = PROMPT_DESCRIPTION.format(intent_name=intent_name, user_utterances="\n".join(utterances))
    chat_completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": content}],
        model=model_name,
    )
    return chat_completion.choices[0].message.content.strip()


async def generate(
    client: AsyncOpenAI, intent_utterances: dict[int, list[str]], intents: list[Intent], model_name: str
) -> list[Intent]:
    tasks = []
    for intent in intents:
        if intent.description is not None:
            continue
        utterances = intent_utterances.get(intent.id, [])
        task = asyncio.create_task(
            generate_intent_description(
                client=client,
                intent_name=intent.name,
                utterances=utterances,
                model_name=model_name,
            )
        )
        tasks.append((intent, task))

    descriptions = await asyncio.gather(*(task for _, task in tasks))
    for (intent, _), description in zip(tasks, descriptions, strict=False):
        intent.description = description
    return intents


def enhance_dataset_with_descriptions(
    dataset: Dataset, api_base: str, api_key: str, model_name: str = "gpt-3.5-turbo"
) -> Dataset:
    client = AsyncOpenAI(
        base_url=api_base,
        api_key=api_key,
    )
    intent_utterances = get_utternaces_by_id(utterances=dataset.utterances)
    dataset.intents = asyncio.run(generate(client, intent_utterances, dataset.intents, model_name))
    return dataset
