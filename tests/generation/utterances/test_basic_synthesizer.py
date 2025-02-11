from unittest.mock import AsyncMock, Mock

from autointent.generation.utterances import SynthesizerChatTemplate, UtteranceGenerator


def has_unfilled_fields(template):
    try:
        # Attempt to format the string with empty values
        template.format(**{})  # noqa: PIE804
        return False  # No unfilled fields  # noqa: TRY300
    except KeyError:
        return True  # Unfilled fields detected


def test_default_chat_template(dataset):
    template = SynthesizerChatTemplate(dataset, split="train_0")
    prompt = template(dataset.intents[0], n_examples=1)
    for msg in prompt:
        assert not has_unfilled_fields(msg["content"])
    assert "extra_instructions" not in prompt


def test_extra_instructions(dataset):
    template = SynthesizerChatTemplate(dataset, split="train_0", extra_instructions="football")
    prompt = template(dataset.intents[0], n_examples=1)[0]["content"]
    assert "extra_instructions" not in prompt
    assert "football" in prompt


def test_on_dataset(dataset):
    mock_llm = Mock()
    mock_llm.get_chat_completion.return_value = "1. LLM answer"

    split_name = "train_0"

    template = SynthesizerChatTemplate(dataset, split=split_name)
    augmenter = UtteranceGenerator(mock_llm, template)

    n_before = len(dataset[split_name])
    new_samples = augmenter.augment(dataset, split_name=split_name, update_split=False)
    n_after = len(dataset[split_name])

    assert n_before == n_after
    assert len(new_samples) == len(dataset.intents)
    assert all(sample.utterance == "LLM answer" for sample in new_samples)

    n_before = len(dataset[split_name])
    new_samples = augmenter.augment(dataset, split_name=split_name, update_split=True)
    n_after = len(dataset[split_name])

    assert n_before + len(new_samples) == n_after
    assert len(new_samples) == len(dataset.intents)


def test_on_dataset_async(dataset):
    mock_llm = AsyncMock()
    mock_llm.get_chat_completion_async.return_value = "1. LLM answer"

    split_name = "train_0"

    template = SynthesizerChatTemplate(dataset, split=split_name)
    augmenter = UtteranceGenerator(mock_llm, template, async_mode=True)

    n_before = len(dataset[split_name])
    new_samples = augmenter.augment(dataset, split_name=split_name, update_split=False)
    n_after = len(dataset[split_name])

    assert n_before == n_after
    assert len(new_samples) == len(dataset.intents)
    assert all(sample.utterance == "LLM answer" for sample in new_samples)

    n_before = len(dataset[split_name])
    new_samples = augmenter.augment(dataset, split_name=split_name, update_split=True)
    n_after = len(dataset[split_name])

    assert n_before + len(new_samples) == n_after
    assert len(new_samples) == len(dataset.intents)


def test_on_dataset_async_with_batch_size(dataset):
    mock_llm = AsyncMock()
    mock_llm.get_chat_completion_async.return_value = "1. LLM answer"

    split_name = "train_0"

    template = SynthesizerChatTemplate(dataset, split=split_name)
    augmenter = UtteranceGenerator(mock_llm, template, async_mode=True)

    batch_size = 2
    new_samples = augmenter.augment(dataset, split_name=split_name, update_split=False, batch_size=batch_size)

    assert len(new_samples) == len(dataset.intents)
    assert all(sample.utterance == "LLM answer" for sample in new_samples)

    batch_size = len(dataset.intents) + 5
    new_samples = augmenter.augment(dataset, split_name=split_name, update_split=False, batch_size=batch_size)

    assert len(new_samples) == len(dataset.intents)
    assert all(sample.utterance == "LLM answer" for sample in new_samples)
