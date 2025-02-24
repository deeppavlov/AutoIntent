import pytest

from autointent import VectorIndex
from autointent.configs import EmbedderConfig


@pytest.fixture
def data_handler():
    class MockDataHandler:
        utterances_train = ["Hello", "How are you", "Goodbye"]  # noqa: RUF012
        labels_train = [0, 1, 2]  # noqa: RUF012

    return MockDataHandler()


def test_create_collection(data_handler):
    vector_index = VectorIndex(embedder_config=EmbedderConfig(model_name="bert-base-uncased"))
    vector_index.add(
        data_handler.utterances_train,
        data_handler.labels_train,
    )
    assert vector_index.embedder.model_name == "bert-base-uncased"
