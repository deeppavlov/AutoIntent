from unittest.mock import Mock

from autointent.generation.utterances import AbstractEvolution, UtteranceEvolver


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
    assert all(sample.utterance == "LLM answer" for sample in new_samples)

    n_before = len(dataset[split_name])
    new_samples = augmenter.augment(dataset, split_name=split_name, n_evolutions=1, update_split=True)
    n_after = len(dataset[split_name])

    assert n_before + len(new_samples) == n_after
    assert len(new_samples) == n_before
