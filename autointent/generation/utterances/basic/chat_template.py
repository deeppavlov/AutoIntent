"""Chat template for evolution augmentation via abstractization."""

import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import ClassVar

from autointent import Dataset
from autointent.generation.utterances.schemas import Message, Role
from autointent.schemas import Intent


class BaseSynthesizer(ABC):
    """Base class."""

    @abstractmethod
    def __call__(self, intent_data: Intent, n_examples: int) -> list[Message]:
        """Generate examples for this intent."""


class BaseSynthesizerChatTemplate(BaseSynthesizer):
    """Base chat template for generating additional examples for a given intent."""

    _MESSAGES_TEMPLATE: ClassVar[list[Message]] = []
    _INTENT_NAME_LABEL: ClassVar[str] = "Intent name"
    _EXAMPLE_UTTERANCES_LABEL: ClassVar[str] = "Example Utterances"
    _GENERATE_INSTRUCTION: ClassVar[str] = "Please generate {n_examples} more examples for the provided intent class.\n"

    def __init__(
        self,
        dataset: Dataset,
        split: str,
        extra_instructions: str | None = None,
        max_sample_utterances: int | None = None,
    ) -> None:
        """Initialize the chat template with dataset, split, and optional instructions."""
        if extra_instructions is None:
            extra_instructions = ""

        self._messages = deepcopy(self._MESSAGES_TEMPLATE)

        if self._messages:
            self._messages[0]["content"] = self._messages[0]["content"].format(extra_instructions=extra_instructions)

        self.dataset = dataset
        self.split = split
        self.max_sample_utterances = max_sample_utterances

    def __call__(self, intent_data: Intent, n_examples: int) -> list[Message]:
        """Generate a list of messages to request additional examples for the given intent."""
        filtered_split = self.dataset[self.split].filter(lambda sample: sample[Dataset.label_feature] == intent_data.id)
        sample_utterances = filtered_split[Dataset.utterance_feature]

        if self.max_sample_utterances is not None and len(sample_utterances) > self.max_sample_utterances:
            sample_utterances = random.sample(sample_utterances, k=self.max_sample_utterances)

        return [
            *self._messages,
            self._create_final_message(intent_data, n_examples, sample_utterances),
        ]

    def _create_final_message(self, intent_data: Intent, n_examples: int, sample_utterances: list[str]) -> Message:
        content = f"{self._INTENT_NAME_LABEL}: {intent_data.name}\n\n" f"{self._EXAMPLE_UTTERANCES_LABEL}:\n"

        if sample_utterances:
            numbered_utterances = "\n".join(f"{i+1}. {utt}" for i, utt in enumerate(sample_utterances))
            content += numbered_utterances + "\n\n"

        content += self._GENERATE_INSTRUCTION.format(n_examples=n_examples)
        return Message(role=Role.USER, content=content)


class SynthesizerChatTemplate(BaseSynthesizerChatTemplate):
    """Chat template for generating additional examples for a given intent class."""

    _MESSAGES_TEMPLATE: ClassVar[list[Message]] = [
        Message(
            role=Role.USER,
            content=(
                "You will be provided with a set of example utterances and the name "
                "of the common topic (intent name) of these utterances. "
                "Your task is to generate more examples that fit within the same intent name.\n\n"
                "Note:\n"
                "- You can generate similar utterances with only slot values changed\n"
                "- You can generate completely different utterance from the same intent name\n"
                "- Intent name can be missed, then you should infer from example utterances only\n"
                "- Example utterances can be missed, then you should infer from intent name only\n"
                "{extra_instructions}\n\n"
                "Intent name: ordering_pizza\n\n"
                "Example Utterances:\n"
                "1. I want to order a large pepperoni pizza.\n"
                "2. Can I get a medium cheese pizza with extra olives?\n"
                "3. Please deliver a small veggie pizza to my address.\n\n"
                "Please generate 3 more examples for the provided intent name."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "1. I'd like to order a large margherita pizza.\n"
                "2. Can you deliver a medium Hawaiian pizza with extra pineapple?\n"
                "3. Please send a small BBQ chicken pizza to my home."
            ),
        ),
        Message(
            role=Role.USER,
            content=(
                "Intent name: booking a hotel\n\n"
                "Example Utterances:\n"
                "1. I need to book a room for two nights in New York.\n\n"
                "Please generate 2 more examples for the provided intent name."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "1. Can you reserve a deluxe room for my trip to Tokyo?\n"
                "2. I need to book a hotel room with a mountain view in Denver."
            ),
        ),
        Message(
            role=Role.USER,
            content=(
                "Intent name:\n\n"
                "Example Utterances:\n"
                "1. What is the weather like today?\n\n"
                "Please generate 2 more examples for the provided intent class."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=("1. Can you tell me the forecast for tomorrow?\n" "2. Is it going to rain this weekend?"),
        ),
        Message(
            role=Role.USER,
            content=(
                "Intent name: Scheduling a Meeting\n\n"
                "Example Utterances:\n\n"
                "Please generate 3 more examples for the provided intent class."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "1. I need to schedule a meeting for next Tuesday.\n"
                "2. Can you set up a conference call for tomorrow afternoon?\n"
                "3. Please arrange a meeting with the marketing team next week."
            ),
        ),
    ]


class SynthesizerChatTemplateRussian(BaseSynthesizerChatTemplate):
    """Russian language template for generating additional intent examples."""

    _MESSAGES_TEMPLATE: ClassVar[list[Message]] = [
        Message(
            role=Role.USER,
            content=(
                "Вам будет предоставлен набор примеров высказываний и название общей темы (интент). "
                "Ваша задача - сгенерировать дополнительные примеры, соответствующие этому интенту.\n\n"
                "Правила:\n"
                "- Можно менять значения слотов в похожих высказываниях\n"
                "- Можно создавать совершенно другие формулировки для того же интента\n"
                "- Если название интента отсутствует, определите его из примеров\n"
                "- Если примеры отсутствуют, используйте только название интента\n"
                "{extra_instructions}\n\n"
                "Название интента: заказ_пиццы\n\n"
                "Примеры высказываний:\n"
                "1. Хочу заказать большую пиццу с пепперони\n"
                "2. Можно среднюю пиццу с сыром и оливками?\n"
                "3. Привезите маленькую вегетарианскую пиццу по моему адресу\n\n"
                "Пожалуйста, сгенерируй еще 3 примера для этого интента."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "1. Мне нужна большая пицца Маргарита\n"
                "2. Можно гавайскую пиццу среднего размера с дополнительным ананасом?\n"
                "3. Доставьте маленькую пиццу с курицей барбекю на дом"
            ),
        ),
        Message(
            role=Role.USER,
            content=(
                "Название интента: бронирование_отеля\n\n"
                "Примеры высказываний:\n"
                "1. Нужно забронировать номер в Москве на две ночи\n\n"
                "Пожалуйста, сгенерируй еще 2 примера для этого интента."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=("1. Забронируйте люкс в Санкт-Петербурге на выходные\n" "2. Ищу номер с видом на море в Сочи"),
        ),
        Message(
            role=Role.USER,
            content=(
                "Название интента:\n\n"
                "Примеры высказываний:\n"
                "1. Какая сегодня погода?\n\n"
                "Пожалуйста, сгенерируй еще 2 примера для этого интента."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=("1. Какой прогноз на завтра?\n" "2. Будет ли дождь в субботу?"),
        ),
        Message(
            role=Role.USER,
            content=(
                "Название интента: запись_на_прием\n\n"
                "Примеры высказываний:\n\n"
                "Пожалуйста, сгенерируй еще 3 примера для этого интента."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "1. Нужно записаться к врачу на следующую неделю\n"
                "2. Хочу назначить встречу с парикмахером на пятницу\n"
                "3. Свободно ли время в пятницу утром для визита?"
            ),
        ),
    ]
    _INTENT_NAME_LABEL = "Название интента"
    _EXAMPLE_UTTERANCES_LABEL = "Примеры высказываний"
    _GENERATE_INSTRUCTION = "Пожалуйста, сгенерируй {n_examples} дополнительных примеров для этого интента.\n"
