from unittest.mock import AsyncMock, Mock

from autointent.generation.utterances import AbstractEvolution, IncrementalUtteranceEvolver, UtteranceEvolver


def test_on_dataset_incremental(dataset):
    mock_llm = Mock()
    mock_llm.get_chat_completion.return_value = "LLM answer"

    split_name = "train_0"

    template = AbstractEvolution()
    augmenter = IncrementalUtteranceEvolver(mock_llm, [template])

    n_before = len(dataset[split_name])
    new_samples = augmenter.augment(dataset, split_name=split_name, n_evolutions=1, update_split=False)
    n_after = len(dataset[split_name])

    assert n_before == n_after
    assert len(new_samples) == n_before
    assert set(new_samples.column_names) == set(dataset[split_name].column_names)

    n_before = len(dataset[split_name])
    new_samples = augmenter.augment(dataset, split_name=split_name, n_evolutions=1, update_split=True)
    n_after = len(dataset[split_name])

    assert n_before + len(new_samples) == n_after
    assert len(new_samples) == n_before
    assert set(new_samples.column_names) == set(dataset[split_name].column_names)


def test_on_dataset_increment_evolver_async(dataset):
    mock_llm = AsyncMock()
    mock_llm.get_chat_completion_async.return_value = "LLM answer"

    split_name = "train_0"

    template = AbstractEvolution()
    augmenter = IncrementalUtteranceEvolver(mock_llm, [template], async_mode=True)

    n_before = len(dataset[split_name])
    new_samples = augmenter.augment(dataset, split_name=split_name, n_evolutions=1, update_split=False)
    n_after = len(dataset[split_name])

    assert n_before == n_after
    assert len(new_samples) == n_before
    assert set(new_samples.column_names) == set(dataset[split_name].column_names)

    n_before = len(dataset[split_name])
    new_samples = augmenter.augment(dataset, split_name=split_name, n_evolutions=1, update_split=True)
    n_after = len(dataset[split_name])

    assert n_before + len(new_samples) == n_after
    assert len(new_samples) == n_before
    assert set(new_samples.column_names) == set(dataset[split_name].column_names)


def test_on_dataset_increment_evolver_async_with_batch_size(dataset):
    mock_llm = AsyncMock()
    mock_llm.get_chat_completion_async.return_value = "LLM answer"

    split_name = "train_0"

    template = AbstractEvolution()
    augmenter = IncrementalUtteranceEvolver(mock_llm, [template], async_mode=True)

    batch_size = 2
    new_samples = augmenter.augment(
        dataset, split_name=split_name, n_evolutions=1, update_split=False, batch_size=batch_size
    )

    assert len(new_samples) == len(dataset[split_name])
    assert set(new_samples.column_names) == set(dataset[split_name].column_names)

    batch_size = len(dataset[split_name]) + 5
    new_samples = augmenter.augment(
        dataset, split_name=split_name, n_evolutions=1, update_split=False, batch_size=batch_size
    )

    assert len(new_samples) == len(dataset[split_name])
    assert set(new_samples.column_names) == set(dataset[split_name].column_names)


def test_default_chat_template(dataset):
    template = AbstractEvolution()
    prompt = template("some utterance", dataset.intents[0])
    assert "some utterance" in prompt[-1]["content"]


def test_on_dataset(dataset):
    mock_llm = Mock()
    mock_llm.get_chat_completion.return_value = "LLM answer"

    split_name = "train_0"

    template = AbstractEvolution()
    augmenter = UtteranceEvolver(mock_llm, [template])

    n_before = len(dataset[split_name])
    new_samples = augmenter.augment(dataset, split_name=split_name, n_evolutions=1, update_split=False)
    n_after = len(dataset[split_name])

    assert n_before == n_after
    assert len(new_samples) == n_before
    assert set(new_samples.column_names) == set(dataset[split_name].column_names)

    n_before = len(dataset[split_name])
    new_samples = augmenter.augment(dataset, split_name=split_name, n_evolutions=1, update_split=True)
    n_after = len(dataset[split_name])

    assert n_before + len(new_samples) == n_after
    assert len(new_samples) == n_before
    assert set(new_samples.column_names) == set(dataset[split_name].column_names)


def test_on_dataset_evolver_async(dataset):
    mock_llm = AsyncMock()
    mock_llm.get_chat_completion_async.return_value = "LLM answer"

    split_name = "train_0"

    template = AbstractEvolution()
    augmenter = UtteranceEvolver(mock_llm, [template], async_mode=True)

    n_before = len(dataset[split_name])
    new_samples = augmenter.augment(dataset, split_name=split_name, n_evolutions=1, update_split=False)
    n_after = len(dataset[split_name])

    assert n_before == n_after
    assert len(new_samples) == n_before
    assert set(new_samples.column_names) == set(dataset[split_name].column_names)

    n_before = len(dataset[split_name])
    new_samples = augmenter.augment(dataset, split_name=split_name, n_evolutions=1, update_split=True)
    n_after = len(dataset[split_name])

    assert n_before + len(new_samples) == n_after
    assert len(new_samples) == n_before
    assert set(new_samples.column_names) == set(dataset[split_name].column_names)


def test_on_dataset_evolver_async_with_batch_size(dataset):
    mock_llm = AsyncMock()
    mock_llm.get_chat_completion_async.return_value = "LLM answer"

    split_name = "train_0"

    template = AbstractEvolution()
    augmenter = UtteranceEvolver(mock_llm, [template], async_mode=True)

    batch_size = 2
    new_samples = augmenter.augment(
        dataset, split_name=split_name, n_evolutions=1, update_split=False, batch_size=batch_size
    )

    assert len(new_samples) == len(dataset[split_name])
    assert set(new_samples.column_names) == set(dataset[split_name].column_names)

    batch_size = len(dataset[split_name]) + 5
    new_samples = augmenter.augment(
        dataset, split_name=split_name, n_evolutions=1, update_split=False, batch_size=batch_size
    )

    assert len(new_samples) == len(dataset[split_name])
    assert set(new_samples.column_names) == set(dataset[split_name].column_names)
