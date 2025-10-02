from unittest.mock import AsyncMock, Mock

from autointent import Sample
from autointent.generation.utterances import CriticHumanLike, HumanUtteranceGenerator


def test_human_utterance_generator_sync(dataset):
    mock_llm = Mock()
    mock_llm.get_chat_completion.return_value = "Human-like utterance"

    mock_critic = Mock(spec=CriticHumanLike)
    mock_critic.is_human.return_value = True

    generator = HumanUtteranceGenerator(mock_llm, mock_critic, async_mode=False)

    n_before = len(dataset["train_0"])
    new_samples = generator.augment(dataset, split_name="train_0", update_split=False, n_final_per_class=2)
    n_after = len(dataset["train_0"])

    assert n_before == n_after
    assert len(new_samples) > 0
    assert all(isinstance(sample, Sample) for sample in new_samples)
    assert all("utterance" in sample.dict() for sample in new_samples)
    assert all("label" in sample.dict() for sample in new_samples)


def test_human_utterance_generator_async(dataset):
    mock_llm = AsyncMock()
    mock_llm.get_chat_completion_async.return_value = "Human-like utterance"

    mock_critic = AsyncMock(spec=CriticHumanLike)
    mock_critic.is_human_async.return_value = True
    generator = HumanUtteranceGenerator(mock_llm, mock_critic, async_mode=True)

    n_before = len(dataset["train_0"])
    new_samples = generator.augment(dataset, split_name="train_0", update_split=False, n_final_per_class=2)
    n_after = len(dataset["train_0"])

    assert n_before == n_after
    assert len(new_samples) > 0
    assert all(isinstance(sample, Sample) for sample in new_samples)
    assert all("utterance" in sample.dict() for sample in new_samples)
    assert all("label" in sample.dict() for sample in new_samples)


def test_human_utterance_generator_respects_critic(dataset):
    mock_llm = Mock()
    mock_llm.get_chat_completion.return_value = "Generated utterance"

    mock_critic = Mock(spec=CriticHumanLike)
    mock_critic.is_human.side_effect = [False, True]

    generator = HumanUtteranceGenerator(mock_llm, mock_critic, async_mode=False)

    new_samples = generator.augment(dataset, split_name="train_0", update_split=False, n_final_per_class=1)
    assert len(new_samples) > 0
    assert all(mock_critic.is_human.call_count >= 1 for _ in range(len(new_samples)))
