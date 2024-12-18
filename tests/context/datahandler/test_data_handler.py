import pytest

from autointent import Dataset
from autointent.context.data_handler import DataHandler
from autointent.schemas import Sample


@pytest.fixture
def sample_multiclass_data():
    return {
        "train": [
            {"utterance": "hello", "label": 0},
            {"utterance": "hi", "label": 0},
            {"utterance": "hey", "label": 0},
            {"utterance": "greetings", "label": 0},
            {"utterance": "what's up", "label": 0},
            {"utterance": "howdy", "label": 0},
            {"utterance": "goodbye", "label": 1},
            {"utterance": "bye", "label": 1},
            {"utterance": "see you later", "label": 1},
            {"utterance": "take care", "label": 1},
            {"utterance": "farewell", "label": 1},
            {"utterance": "catch you later", "label": 1},
        ],
        "test": [
            {"utterance": "greetings", "label": 0},
            {"utterance": "farewell", "label": 1},
        ],
        "intents": [
            {
                "id": 0,
                "regexp_full_match": [r"^(hello|hi)$"],
                "regexp_partial_match": [r"(hello|hi)"],
            },
            {
                "id": 1,
                "regexp_full_match": [r"^(goodbye|bye)$"],
                "regexp_partial_match": [r"(goodbye|bye)"],
            },
        ],
    }


@pytest.fixture
def sample_multilabel_data():
    return {
        "train": [
            {"utterance": "hello and goodbye", "label": [0, 1]},
            {"utterance": "hi there", "label": [0]},
            {"utterance": "farewell and see you later", "label": [1]},
            {"utterance": "good morning", "label": [0]},
            {"utterance": "goodbye for now", "label": [1]},
            {"utterance": "hey, how's it going?", "label": [0]},
            {"utterance": "so long and take care", "label": [1]},
            {"utterance": "hello, nice to meet you", "label": [0]},
            {"utterance": "bye, have a great day", "label": [1]},
            {"utterance": "what's up?", "label": [0]},
            {"utterance": "later, see you soon", "label": [1]},
            {"utterance": "greetings and salutations", "label": [0]},
        ],
        "test": [
            {"utterance": "greetings", "label": [0]},
            {"utterance": "farewell", "label": [1]},
        ],
    }


def mock_split():
    return [{"utterance": "Hello!", "label": 0}]


def test_data_handler_initialization(sample_multiclass_data):
    handler = DataHandler(dataset=Dataset.from_dict(sample_multiclass_data), random_seed=42)

    assert handler.multilabel is False
    assert handler.n_classes == 2
    assert handler.train_utterances(0) == ["hello", "bye", "hi", "take care"]
    assert handler.test_utterances() == ["greetings", "farewell"]
    assert handler.train_labels(0) == [0, 1, 0, 1]
    assert handler.test_labels() == [0, 1]


def test_data_handler_multilabel_mode(sample_multilabel_data):
    handler = DataHandler(dataset=Dataset.from_dict(sample_multilabel_data), random_seed=42)

    assert handler.multilabel is True
    assert handler.n_classes == 2
    assert handler.train_utterances(0) == [
        "so long and take care",
        "what's up?",
        "later, see you soon",
        "greetings and salutations",
    ]
    assert handler.test_utterances() == ["greetings", "farewell"]
    assert handler.train_labels(0) == [[0, 1], [1, 0], [0, 1], [1, 0]]
    assert handler.test_labels() == [[1, 0], [0, 1]]


@pytest.mark.parametrize("label", [0, [0], None])
def test_sample_initialization(label):
    sample = Sample(utterance="Hello!", label=label)
    assert sample.label == label


@pytest.mark.parametrize("label", [-1, [-1], []])
def test_sample_validation(label):
    with pytest.raises(ValueError):
        Sample(utterance="Hello!", label=label)


@pytest.mark.parametrize(
    "mapping",
    [
        {"train": mock_split()},
        {"train": mock_split(), "test": mock_split()},
        {"train_0": mock_split(), "train_1": mock_split()},
        {"train_0": mock_split(), "train_1": mock_split(), "test": mock_split()},
        {"train": mock_split(), "validation": mock_split()},
        {"train": mock_split(), "validation": mock_split(), "test": mock_split()},
        {"train": mock_split(), "validation_0": mock_split(), "validation_1": mock_split()},
        {"train": mock_split(), "validation_0": mock_split(), "validation_1": mock_split(), "test": mock_split()},
        {"train_0": mock_split(), "train_1": mock_split(), "validation": mock_split()},
        {"train_0": mock_split(), "train_1": mock_split(), "validation": mock_split(), "test": mock_split()},
        {"train_0": mock_split(), "train_1": mock_split(), "validation_0": mock_split(), "validation_1": mock_split()},
        {
            "train_0": mock_split(),
            "train_1": mock_split(),
            "validation_0": mock_split(),
            "validation_1": mock_split(),
            "test": mock_split(),
        },
    ]
)
def test_dataset_initialization(mapping):
    dataset = Dataset.from_dict(mapping)
    for split in mapping:
        assert split in dataset


@pytest.mark.parametrize(
    "mapping",
    [
        {},
        {"train_0": mock_split()},
        {"train_1": mock_split()},
        {"train": mock_split(), "train_0": mock_split()},
        {"train": mock_split(), "train_1": mock_split()},
        {"train": mock_split(), "train_0": mock_split(), "train_1": mock_split()},
        {"train": mock_split(), "validation_0": mock_split()},
        {"train": mock_split(), "validation_1": mock_split()},
        {"train": mock_split(), "validation": mock_split(), "validation_0": mock_split()},
        {"train": mock_split(), "validation": mock_split(), "validation_1": mock_split()},
        {"train": mock_split(), "validation": mock_split(), "validation_0": mock_split(), "validation_1": mock_split()},
        {"train": mock_split(), "oos": mock_split()}
    ]
)
def test_dataset_validation(mapping):
    with pytest.raises(ValueError):
        Dataset.from_dict(mapping)


@pytest.mark.parametrize(
    "mapping",
    [
        {"train": [{"utterance": "Hello!", "label": 0}], "intents": [{"id": 1}]},
        {"train": [{"utterance": "Hello!", "label": 0}, {"utterance": "Hello!", "label": 1}], "intents": [{"id": 0}]},
        {
            "train": [{"utterance": "Hello!", "label": 0}, {"utterance": "Hello!", "label": 1}],
            "test": [{"utterance": "Hello!", "label": 0}],
        },
        {"train": [{"utterance": "Hello!"}]},
    ]
)
def test_intents_validation(mapping):
    with pytest.raises(ValueError):
        Dataset.from_dict(mapping)
