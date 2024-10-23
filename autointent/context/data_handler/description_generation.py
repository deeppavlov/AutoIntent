from collections import defaultdict

from openai import OpenAI

from autointent.context.data_handler.schemas import Dataset, Utterance

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
                intent_utterances[label].append(utterance.text)

    return intent_utterances


def generate_intent_description(client: OpenAI, intent_name: str, utterances: list[str], model_name: str) -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": PROMPT_DESCRIPTION.format(intent_name=intent_name, utterances="\n".join(utterances)),
            }
        ],
        model=model_name,
    )
    return chat_completion.choices[0].text.strip()


def enhance_dataset_with_descriptions(
    dataset: Dataset, api_base: str, api_key: str, model_name: str = "gpt-3.5-turbo"
) -> Dataset:
    client = OpenAI(
        api_base=api_base,
        api_key=api_key,
    )
    intent_utterances = get_utternaces_by_id(utterances=dataset.utterances)

    for intent in dataset.intents:
        if not intent.description:
            intent.description = generate_intent_description(
                client=client,
                intent_name=intent.name,
                utterances=intent_utterances.get(intent.id, []),
                model_name=model_name,
            )
    return dataset
