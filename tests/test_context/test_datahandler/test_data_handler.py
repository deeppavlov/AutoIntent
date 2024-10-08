import pytest

from autointent.context.data_handler import DataHandler


@pytest.fixture
def sample_multiclass_intent_records():
    return [
        {
            "intent_id": 0,
            "sample_utterances": ["hello", "hi"],
            "regexp_full_match": [r"^(hello|hi)$"],
            "regexp_partial_match": [r"(hello|hi)"],
        },
        {
            "intent_id": 1,
            "sample_utterances": ["goodbye", "bye"],
            "regexp_full_match": [r"^(goodbye|bye)$"],
            "regexp_partial_match": [r"(goodbye|bye)"],
        },
    ]


@pytest.fixture
def sample_multilabel_utterance_records():
    return [
        {"utterance": "hello and goodbye", "labels": [0, 1]},
        {"utterance": "hi there", "labels": [0]},
    ]


@pytest.fixture
def sample_test_utterance_records():
    return [
        {"utterance": "greetings", "labels": [0]},
        {"utterance": "farewell", "labels": [1]},
    ]


def test_data_handler_initialization(
    sample_multiclass_intent_records, sample_multilabel_utterance_records, sample_test_utterance_records
):
    handler = DataHandler(
        sample_multiclass_intent_records,
        sample_multilabel_utterance_records,
        sample_test_utterance_records,
        mode="multiclass",
        seed=42,
    )

    assert handler.multilabel is False
    assert handler.n_classes == 2
    assert handler.utterances_train == ["hello", "hi", "goodbye", "bye"]
    assert handler.utterances_test == ["greetings", "farewell"]
    assert handler.labels_train == [0, 0, 1, 1]
    assert handler.labels_test == [0, 1]


def test_data_handler_multilabel_mode(
    sample_multiclass_intent_records, sample_multilabel_utterance_records, sample_test_utterance_records
):
    handler = DataHandler(
        sample_multiclass_intent_records,
        sample_multilabel_utterance_records,
        sample_test_utterance_records,
        mode="multilabel",
        seed=42,
    )

    assert handler.multilabel is True
    assert handler.n_classes == 2
    assert handler.utterances_train == ["hello and goodbye", "hi there"]
    assert handler.utterances_test == ["greetings", "farewell"]
    assert handler.labels_train == [[1, 1], [1, 0]]
    assert handler.labels_test == [[1, 0], [0, 1]]


def test_regex_sampling(
    sample_multiclass_intent_records, sample_multilabel_utterance_records, sample_test_utterance_records
):
    sampling = 5
    data = DataHandler(
        sample_multiclass_intent_records,
        sample_multilabel_utterance_records,
        sample_test_utterance_records,
        mode="multiclass",
        regex_sampling=sampling,
        seed=42,
    )
    assert data.utterances_train == ["hello", "hi", "hello", "hi", "goodbye", "bye", "goodbye", "bye"]


def test_has_oos_samples(
    sample_multiclass_intent_records, sample_multilabel_utterance_records, sample_test_utterance_records
):
    handler = DataHandler(
        sample_multiclass_intent_records,
        sample_multilabel_utterance_records,
        sample_test_utterance_records,
        mode="multiclass",
        seed=42,
    )

    assert isinstance(handler.has_oos_samples(), bool)


def test_dump_method(
    sample_multiclass_intent_records, sample_multilabel_utterance_records, sample_test_utterance_records
):
    handler = DataHandler(
        sample_multiclass_intent_records,
        sample_multilabel_utterance_records,
        sample_test_utterance_records,
        mode="multiclass",
        seed=42,
    )

    train_data, test_data = handler.dump()

    assert isinstance(train_data, list)
    assert isinstance(test_data, list)
    assert len(train_data) > 0
    assert len(test_data) > 0


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
