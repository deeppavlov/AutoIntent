import numpy as np
import pytest
import torch

from autointent import Dataset
from autointent.modules.scoring import GCNScorer


@pytest.fixture
def multilabel_dataset():
    data = {
        "train": [
            {"utterance": "utterance 1", "label": [1, 0, 0]},
            {"utterance": "utterance 2", "label": [0, 1, 0]},
            {"utterance": "utterance 3", "label": [0, 0, 1]},
            {"utterance": "utterance 4", "label": [1, 1, 0]},
        ],
        "intents": [
            {"id": 0, "name": "intent_0"},
            {"id": 1, "name": "intent_1"},
            {"id": 2, "name": "intent_2"},
        ],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def multiclass_dataset():
    data = {
        "train": [
            {"utterance": "utterance 1", "label": 0},
            {"utterance": "utterance 2", "label": 1},
            {"utterance": "utterance 3", "label": 2},
            {"utterance": "utterance 4", "label": 0},
        ],
        "intents": [
            {"id": 0, "name": "intent_0"},
            {"id": 1, "name": "intent_1"},
            {"id": 2, "name": "intent_2"},
        ],
    }
    return Dataset.from_dict(data)


def test_gcn_scorer_multilabel(multilabel_dataset):
    torch.manual_seed(42)
    scorer = GCNScorer(embedder_config="prajjwal1/bert-tiny", num_train_epochs=1, batch_size=2, seed=42)
    train_utterances = multilabel_dataset["train"]["utterance"]
    train_labels = multilabel_dataset["train"]["label"]
    descriptions = [intent.name for intent in multilabel_dataset.intents]

    scorer.fit(train_utterances, train_labels, descriptions)
    test_utterances = ["test 1", "test 2"]
    predictions = scorer.predict(test_utterances)

    expected_predictions = np.array([[0.481326, 0.480452, 0.487172], [0.481336, 0.480337, 0.487094]])
    np.testing.assert_allclose(predictions, expected_predictions, atol=1e-5)


def test_gcn_scorer_multiclass(multiclass_dataset):
    torch.manual_seed(42)
    scorer = GCNScorer(embedder_config="prajjwal1/bert-tiny", num_train_epochs=1, batch_size=2, seed=42)
    train_utterances = multiclass_dataset["train"]["utterance"]
    train_labels = multiclass_dataset["train"]["label"]
    descriptions = [intent.name for intent in multiclass_dataset.intents]

    scorer.fit(train_utterances, train_labels, descriptions)
    test_utterances = ["test 1", "test 2"]
    predictions = scorer.predict(test_utterances)

    expected_predictions = np.array([[0.345223, 0.322953, 0.331824], [0.345442, 0.322777, 0.331781]])
    np.testing.assert_allclose(predictions, expected_predictions, atol=1e-5)
    np.testing.assert_allclose(predictions.sum(axis=1), 1.0, atol=1e-6)


def test_gcn_scorer_dump_load(tmp_path, multilabel_dataset):
    torch.manual_seed(42)
    scorer = GCNScorer(embedder_config="prajjwal1/bert-tiny", num_train_epochs=1, batch_size=2, seed=42)
    train_utterances = multilabel_dataset["train"]["utterance"]
    train_labels = multilabel_dataset["train"]["label"]
    descriptions = [intent.name for intent in multilabel_dataset.intents]
    scorer.fit(train_utterances, train_labels, descriptions)

    test_utterances = ["test utterance 1"]
    original_predictions = scorer.predict(test_utterances)

    scorer.dump(str(tmp_path))

    loaded_scorer = GCNScorer.load(str(tmp_path))
    loaded_predictions = loaded_scorer.predict(test_utterances)

    np.testing.assert_allclose(original_predictions, loaded_predictions, atol=1e-6)
