import asyncio
from collections import defaultdict

from openai import AsyncOpenAI

from autointent.context.data_handler.schemas import Dataset, Intent, Utterance

PROMPT_DESCRIPTION = """
{intent_name}
{utterances}
"""
# где лучше использовать PROMPT_DESCRIPTION? вставить его в функцию или вынести куда-то?
# куда внедрить enhance_dataset_with_description? в datahandler? задать булевый аргумент в гидре добавлять\не добавлять описание?
# по остальным аргументами api_base, api_key, model_name тоже добавить в гидру?


def get_utternaces_by_id(utterances: list[Utterance]) -> dict[int, list[str]]:
    intent_utterances = defaultdict(list)

    for utterance in utterances:
        if utterance.label is not None:
            for label in utterance.label:
                text = utterance.text
                intent_utterances[label].append(text)

    return intent_utterances


async def generate_intent_description(
    client: AsyncOpenAI, intent_name: str, utterances: list[str], model_name: str
) -> str:
    content = PROMPT_DESCRIPTION.format(intent_name=intent_name, utterances="\n".join(utterances))
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
        api_base=api_base,
        api_key=api_key,
    )
    intent_utterances = get_utternaces_by_id(utterances=dataset.utterances)
    dataset.intents = asyncio.run(generate(client, intent_utterances, dataset.intents, model_name))
    return dataset
