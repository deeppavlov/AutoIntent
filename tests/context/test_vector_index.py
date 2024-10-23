import pytest

from autointent.context.vector_index_client import VectorIndexClient


@pytest.fixture
def data_handler():
    class MockDataHandler:
        utterances_train = ["Hello", "How are you", "Goodbye"]  # noqa: RUF012
        labels_train = [0, 1, 2]  # noqa: RUF012

    return MockDataHandler()


def test_vector_index_initialization(setup_environment):
    db_dir, dump_dir, logs_dir = setup_environment
    vector_index_client = VectorIndexClient("cpu", db_dir)
    assert vector_index_client.device == "cpu"


def test_create_collection(data_handler, setup_environment):
    db_dir, dump_dir, logs_dir = setup_environment
    vector_index_client = VectorIndexClient("cpu", db_dir)
    vector_index = vector_index_client.create_index("bert-base-uncased", data_handler)
    assert vector_index.model_name == "bert-base-uncased"
