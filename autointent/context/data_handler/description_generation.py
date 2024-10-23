from openai import OpenAI

from autointent.context.data_handler.schemas import Dataset, Intent, Utterance

PROMPT_DESCRIPTION = """
{intent_name}
{utterances}
"""
# где лучше использовать PROMPT_DESCRIPTION? вставить его в функцию или вынести куда-то?
# куда внедрить enhance_dataset_with_description? в datahandler? задать булевый аргумент в гидре добавлять\не добавлять описание?
# по остальным аргументами api_base, api_key, model_name тоже добавить в гидру?


def get_utternaces_by_id(intents: list[Intent], utterances: list[Utterance]) -> dict[int, list[str]]:
    pass


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
    intent_utterances = get_utternaces_by_id(dataset.intents, dataset.utterances)

    for intent in dataset.intents:
        if not intent.description:
            intent.description = generate_intent_description(
                client=client, intent_name=intent.name, utterances=intent_utterances[intent.id], model_name=model_name
            )
    return dataset
