import pytest

from autointent.context.data_handler import DataAugmenter, DataHandler, Dataset


@pytest.fixture
def sample_multiclass_data():
    return {
        "train": [
            {"utterance": "hello", "label": 0},
            {"utterance": "hi", "label": 0},
            {"utterance": "goodbye", "label": 1},
            {"utterance": "bye", "label": 1},
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
        ],
        "test": [
            {"utterance": "greetings", "label": [0]},
            {"utterance": "farewell", "label": [1]},
        ],
    }


def test_data_handler_initialization(sample_multiclass_data):
    handler = DataHandler(dataset=Dataset.load_dict(sample_multiclass_data), random_seed=42)

    assert handler.multilabel is False
    assert handler.n_classes == 2
    assert handler.train_utterances == ["hello", "hi", "goodbye", "bye"]
    assert handler.test_utterances == ["greetings", "farewell"]
    assert handler.train_labels == [0, 0, 1, 1]
    assert handler.test_labels == [0, 1]


def test_data_handler_multilabel_mode(sample_multilabel_data):
    handler = DataHandler(dataset=Dataset.load_dict(sample_multilabel_data), random_seed=42)

    assert handler.multilabel is True
    assert handler.n_classes == 2
    assert handler.train_utterances == ["hello and goodbye", "hi there"]
    assert handler.test_utterances == ["greetings", "farewell"]
    assert handler.train_labels == [[1, 1], [1, 0]]
    assert handler.test_labels == [[1, 0], [0, 1]]


def test_regex_sampling(sample_multiclass_data):
    train_data, test_data = sample_multiclass_data
    augmenter = DataAugmenter(regex_sampling=5, random_seed=42)
    dataset = Dataset.model_validate(train_data)
    augmented_dataset = augmenter(dataset)
    utterances_train = [ut.text for ut in augmented_dataset.utterances]
    assert utterances_train == ["hello", "hi", "goodbye", "bye", "hello", "hi", "goodbye", "bye"]


def test_dump_method(sample_multiclass_data):
    train_data, test_data = sample_multiclass_data
    handler = DataHandler(
        dataset=Dataset.model_validate(train_data),
        test_dataset=Dataset.model_validate(test_data),
        random_seed=42,
    )

    train_data, test_data = handler.dump()

    assert train_data == [
        {"intent_id": 0, "sample_utterances": ["hello", "hi"]},
        {"intent_id": 1, "sample_utterances": ["goodbye", "bye"]},
    ]
    assert test_data == [{"labels": [0], "utterance": "greetings"}, {"labels": [1], "utterance": "farewell"}]


@pytest.mark.skip("All data validations will be refactored later")
def test_error_handling(
    sample_multiclass_intent_records, sample_multilabel_utterance_records, sample_test_utterance_records
):
    with pytest.raises(ValueError, match="unexpected classification mode value"):
        DataHandler(
            sample_multiclass_intent_records,
            sample_multilabel_utterance_records,
            sample_test_utterance_records,
            mode="invalid_mode",
            seed=42,
        )

    with pytest.raises(ValueError, match="No data provided"):
        DataHandler([], [], [], mode="multiclass", seed=42)
