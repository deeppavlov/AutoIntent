import pytest

from autointent.context.vector_index import VectorIndex


@pytest.fixture
def data_handler():
    class MockDataHandler:
        utterances_train = ["Hello", "How are you", "Goodbye"]  # noqa: RUF012
        labels_train = [0, 1, 2]  # noqa: RUF012

    return MockDataHandler()


def test_vector_index_initialization():
    vector_index = VectorIndex("cpu", False, 3)
    assert vector_index.device == "cpu"
    assert vector_index.multilabel is False
    assert vector_index.n_classes == 3


def test_create_collection(data_handler):
    vector_index = VectorIndex("cpu", False, 3)
    collection = vector_index.create_index("bert-base-uncased", data_handler)
    assert collection == "bert-base-uncased"


def test_metadata_as_labels_multiclass():
    vector_index = VectorIndex("cpu", False, 3)
    metadata = [{"intent_id": 0}, {"intent_id": 1}, {"intent_id": 2}]
    labels = vector_index.metadata_as_labels(metadata)
    assert labels == [0, 1, 2]


def test_metadata_as_labels_multilabel():
    vector_index = VectorIndex("cpu", True, 3)
    metadata = [{"0": 1, "1": 0, "2": 1}, {"0": 0, "1": 1, "2": 1}]
    labels = vector_index.metadata_as_labels(metadata)
    assert labels == [[1, 0, 1], [0, 1, 1]]


def test_labels_as_metadata_multiclass():
    vector_index = VectorIndex("cpu", False, 3)
    labels = [0, 1, 2]
    metadata = vector_index.labels_as_metadata(labels)
    assert metadata == [{"intent_id": 0}, {"intent_id": 1}, {"intent_id": 2}]


def test_labels_as_metadata_multilabel():
    vector_index = VectorIndex("cpu", True, 3)
    labels = [[1, 0, 1], [0, 1, 1]]
    metadata = vector_index.labels_as_metadata(labels)
    assert metadata == [{"0": 1, "1": 0, "2": 1}, {"0": 0, "1": 1, "2": 1}]
