import numpy as np
import pytest

from autointent.context.data_handler.stratification import (
    get_sample_utterances,
    is_multiclass_test_set_complete,
    is_multilabel_test_set_complete,
    multilabel_train_test_split,
    to_onehot,
    validate_test_labels,
)


@pytest.fixture
def sample_intent_records():
    return [
        {"intent_id": 0, "sample_utterances": ["hello", "hi"]},
        {"intent_id": 1, "sample_utterances": ["goodbye", "bye"]},
        {"intent_id": -1, "sample_utterances": ["unknown1", "unknown2"]},  # OOS samples
    ]


@pytest.fixture
def sample_multilabel_records():
    return [
        {"utterance": "hello goodbye", "labels": [0, 1]},
        {"utterance": "hi there", "labels": [0]},
        {"utterance": "see you", "labels": [1]},
        {"utterance": "unknown", "labels": []},  # OOS sample
    ]


def test_get_sample_utterances(sample_intent_records):
    utterances, labels = get_sample_utterances(sample_intent_records)
    assert utterances == ["hello", "hi", "goodbye", "bye", "unknown1", "unknown2"]
    assert labels == [0, 0, 1, 1, -1, -1]


def test_multilabel_train_test_split():
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

    x_train, x_test, y_train, y_test = multilabel_train_test_split(x, y, test_size=0.5, random_state=42, stratify=y)

    assert x_train.shape == (2, 2)
    assert x_test.shape == (2, 2)
    assert y_train.shape == (2, 2)
    assert y_test.shape == (2, 2)
    assert np.array_equal(y_train.sum(axis=0), y_test.sum(axis=0))  # Check stratification


def test_validate_test_labels():
    assert validate_test_labels([0, 1, 2], False, 3)
    assert validate_test_labels([[1, 0, 0], [0, 1, 0], [0, 0, 1]], True, 3)

    with pytest.raises(ValueError, match="unexpected labels format"):
        validate_test_labels([0, 1], True, 3)


def test_is_multilabel_test_set_complete():
    assert is_multilabel_test_set_complete(np.array([[1, 0], [0, 1]]))
    assert not is_multilabel_test_set_complete(np.array([[1, 0], [1, 0]]))


def test_is_multiclass_test_set_complete():
    assert is_multiclass_test_set_complete([0, 1, 2], 3)
    assert not is_multiclass_test_set_complete([0, 1, 1], 3)


def test_to_onehot():
    labels = np.array([0, 1, 2])
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.array_equal(to_onehot(labels, 3), expected)
