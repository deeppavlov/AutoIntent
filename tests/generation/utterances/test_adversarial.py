from unittest.mock import AsyncMock, Mock

import pytest

from autointent import Dataset
from autointent.generation.utterances import CriticHumanLike, HumanUtteranceGenerator
from autointent.schemas import Sample


@pytest.fixture
def dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "intents": [
                {"id": 0, "name": "Greeting"},
                {"id": 1, "name": "OrderFood"},
            ],
            "train": [
                {"utterance": "hello", "label": 0},
                {"utterance": "hi there", "label": 0},
                {"utterance": "i want pizza", "label": 1},
            ],
        }
    )


def test_human_utterance_generator_sync(dataset: Dataset) -> None:
    mock_llm = Mock()
    mock_llm.get_chat_completion.return_value = "Human-like utterance"

    mock_critic = Mock(spec=CriticHumanLike)
    mock_critic.is_human.return_value = True

    generator = HumanUtteranceGenerator(mock_llm, mock_critic, async_mode=False)

    n_before = len(dataset["train"])
    new_samples = generator.augment(dataset, split_name="train", update_split=False, n_final_per_class=2)
    n_after = len(dataset["train"])

    assert n_before == n_after
    assert len(new_samples) > 0
    assert all(isinstance(sample, Sample) for sample in new_samples)
    assert all("utterance" in sample.dict() for sample in new_samples)
    assert all("label" in sample.dict() for sample in new_samples)


def test_human_utterance_generator_async(dataset: Dataset) -> None:
    mock_llm = AsyncMock()
    mock_llm.get_chat_completion_async.return_value = "Human-like utterance"

    mock_critic = AsyncMock(spec=CriticHumanLike)
    mock_critic.is_human_async.return_value = True

    generator = HumanUtteranceGenerator(mock_llm, mock_critic, async_mode=True)

    n_before = len(dataset["train"])
    new_samples = generator.augment(dataset, split_name="train", update_split=False, n_final_per_class=2)
    n_after = len(dataset["train"])
    assert n_before == n_after
    assert len(new_samples) > 0
    assert all(isinstance(sample, Sample) for sample in new_samples)
    assert all("utterance" in sample.dict() for sample in new_samples)
    assert all("label" in sample.dict() for sample in new_samples)


def test_human_utterance_generator_respects_critic(dataset: Dataset) -> None:
    mock_llm = Mock()
    mock_llm.get_chat_completion.return_value = "Generated utterance"

    mock_critic = Mock(spec=CriticHumanLike)
    mock_critic.is_human.return_value = True
    generator = HumanUtteranceGenerator(mock_llm, mock_critic, async_mode=False)
    new_samples = generator.augment(dataset, split_name="train", update_split=False, n_final_per_class=1)
    assert len(new_samples) > 0
    assert mock_critic.is_human.call_count >= 1
