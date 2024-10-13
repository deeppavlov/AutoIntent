import pytest

from autointent.context.vector_index import VectorIndex


@pytest.fixture
def data_handler():
    class MockDataHandler:
        utterances_train = ["Hello", "How are you", "Goodbye"]  # noqa: RUF012
        labels_train = [0, 1, 2]  # noqa: RUF012

    return MockDataHandler()


def test_vector_index_initialization(tmp_path):
    vector_index = VectorIndex("cpu", False, 3)
    assert vector_index.device == "cpu"
    assert vector_index.multilabel is False
    assert vector_index.n_classes == 3


def test_get_collection(tmp_path):
    vector_index = VectorIndex("cpu", False, 3)
    collection = vector_index.get_index("bert-base-uncased")
    assert collection.name == "bert-base-uncased"


def test_create_collection(tmp_path, data_handler):
    vector_index = VectorIndex("cpu", False, 3)
    collection = vector_index.create_index("bert-base-uncased", data_handler)
    assert isinstance(collection, Collection)


def test_delete_collection(tmp_path):
    vector_index = VectorIndex("cpu", False, 3)
    vector_index.get_collection("bert-base-uncased")  # Create collection
    vector_index.delete_collection("bert-base-uncased")
    assert "bert-base-uncased" not in vector_index.client.list_collections()


def test_metadata_as_labels_multiclass(tmp_path):
    vector_index = VectorIndex("cpu", False, 3)
    metadata = [{"intent_id": 0}, {"intent_id": 1}, {"intent_id": 2}]
    labels = vector_index.metadata_as_labels(metadata)
    assert labels == [0, 1, 2]


def test_metadata_as_labels_multilabel(tmp_path):
    vector_index = VectorIndex("cpu", True, 3)
    metadata = [{"0": 1, "1": 0, "2": 1}, {"0": 0, "1": 1, "2": 1}]
    labels = vector_index.metadata_as_labels(metadata)
    assert labels == [[1, 0, 1], [0, 1, 1]]


def test_labels_as_metadata_multiclass(tmp_path):
    vector_index = VectorIndex("cpu", False, 3)
    labels = [0, 1, 2]
    metadata = vector_index.labels_as_metadata(labels)
    assert metadata == [{"intent_id": 0}, {"intent_id": 1}, {"intent_id": 2}]


def test_labels_as_metadata_multilabel(tmp_path):
    vector_index = VectorIndex("cpu", True, 3)
    labels = [[1, 0, 1], [0, 1, 1]]
    metadata = vector_index.labels_as_metadata(labels)
    assert metadata == [{"0": 1, "1": 0, "2": 1}, {"0": 0, "1": 1, "2": 1}]
