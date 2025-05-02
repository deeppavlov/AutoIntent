from collections import Counter

import pytest

from autointent import Dataset
from autointent.configs import DataConfig
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
                "regex_full_match": [r"^(hello|hi)$"],
                "regex_partial_match": [r"(hello|hi)"],
            },
            {
                "id": 1,
                "regex_full_match": [r"^(goodbye|bye)$"],
                "regex_partial_match": [r"(goodbye|bye)"],
            },
        ],
    }


@pytest.fixture
def sample_multilabel_data():
    return {
        "train": [
            {"utterance": "hello and goodbye", "label": [0, 1]},
            {"utterance": "hi there", "label": [0, 1]},
            {"utterance": "farewell and see you later", "label": [0, 1]},
            {"utterance": "good morning", "label": [1, 0]},
            {"utterance": "goodbye for now", "label": [1, 0]},
            {"utterance": "hey, how's it going?", "label": [1, 0]},
            {"utterance": "so long and take care", "label": [0, 1]},
            {"utterance": "hello, nice to meet you", "label": [0, 1]},
            {"utterance": "bye, have a great day", "label": [0, 1]},
            {"utterance": "what's up?", "label": [1, 0]},
            {"utterance": "later, see you soon", "label": [1, 0]},
            {"utterance": "greetings and salutations", "label": [1, 0]},
        ],
        "test": [
            {"utterance": "greetings", "label": [0, 1]},
            {"utterance": "farewell", "label": [1, 0]},
        ],
    }


def mock_split():
    return [{"utterance": "Hello!", "label": 0}]


def test_data_handler_initialization(sample_multiclass_data):
    handler = DataHandler(dataset=Dataset.from_dict(sample_multiclass_data), random_seed=42)

    assert handler.multilabel is False
    assert handler.dataset.n_classes == 2
    assert handler.train_utterances(0) == ["hello", "bye", "hi", "take care"]
    assert handler.test_utterances() == ["greetings", "farewell"]
    assert handler.train_labels(0) == [0, 1, 0, 1]
    assert handler.test_labels() == [0, 1]


def test_data_handler_multilabel_mode(sample_multilabel_data):
    handler = DataHandler(dataset=Dataset.from_dict(sample_multilabel_data), random_seed=42)

    assert handler.multilabel is True
    assert handler.dataset.n_classes == 2
    assert handler.train_utterances(0) == [
        "hey, how's it going?",
        "so long and take care",
        "hello, nice to meet you",
        "later, see you soon",
    ]
    assert handler.test_utterances() == ["greetings", "farewell"]
    assert handler.train_labels(0) == [[1, 0], [0, 1], [0, 1], [1, 0]]
    assert handler.test_labels() == [[0, 1], [1, 0]]


@pytest.mark.parametrize("label", [0, [0, 1, 0], None])
def test_sample_initialization(label):
    sample = Sample(utterance="Hello!", label=label)
    assert sample.label == label


@pytest.mark.parametrize("label", [-1, [-1], []])
def test_sample_validation(label):
    with pytest.raises(ValueError):  # noqa: PT011
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
    ],
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
    ],
)
def test_dataset_validation(mapping):
    with pytest.raises(ValueError):  # noqa: PT011
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
    ],
)
def test_intents_validation(mapping):
    with pytest.raises(ValueError):  # noqa: PT011
        Dataset.from_dict(mapping)


def count_oos(split):
    return len(split.filter(lambda sample: sample["label"] is None))


def test_cv_folding(dataset):
    DataHandler(dataset, config=DataConfig(scheme="cv", n_folds=3))

    desired_specs = {
        "test": {"total": 12, "oos": 4},
        "train_0": {"total": 16, "oos": 5},
        "train_1": {"total": 16, "oos": 5},
        "train_2": {"total": 16, "oos": 6},
    }

    for split_name in dataset:
        assert len(dataset[split_name]) == desired_specs[split_name]["total"]
        assert count_oos(dataset[split_name]) == desired_specs[split_name]["oos"]


def count_oos_labels(split):
    return sum(sample is None for sample in split)


def test_cv_iterator(dataset):
    dh = DataHandler(dataset, config=DataConfig(scheme="cv", n_folds=3))

    desired_specs = [
        {
            "train": {"total": 21, "oos": 0},
            "val": {"total": 16, "oos": 5},
        },
        {
            "train": {"total": 21, "oos": 0},
            "val": {"total": 16, "oos": 5},
        },
        {
            "train": {"total": 22, "oos": 0},
            "val": {"total": 16, "oos": 6},
        },
    ]

    for i, (x_train, y_train, x_val, y_val) in enumerate(dh.validation_iterator()):
        specs = desired_specs[i]
        assert len(x_train) == len(y_train) == specs["train"]["total"]
        assert count_oos_labels(y_train) == specs["train"]["oos"]
        assert len(x_val) == len(y_val) == specs["val"]["total"]
        assert count_oos_labels(y_val) == specs["val"]["oos"]


def test_few_shot_split(dataset):
    dh = DataHandler(dataset, config=DataConfig(scheme="ho", is_few_shot_train=True, examples_per_intent=2))

    desired_specs = {
        "train_0": {0: 2, 1: 2, 2: 2, 3: 2},
        "train_1": {2: 2, 0: 2, None: 2, 1: 1, 3: 1},
        "validation_0": {0: 3, 1: 4, 2: 3, 3: 4},
        "validation_1": {None: 14, 3: 1, 0: 1, 1: 1, 2: 1},
        "test": {None: 4, 0: 2, 2: 2, 3: 2, 1: 2},
    }

    for data_split in dh.dataset:
        assert (
            Counter(dh.dataset[data_split][dh.dataset.label_feature]) == desired_specs[data_split]
        ), f"Failed for {data_split}"
