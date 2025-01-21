from unittest.mock import Mock

import pytest

from autointent import VectorIndex
from autointent.context.data_handler import DataHandler


@pytest.fixture
def mock_data_handler():
    mock = Mock(spec=DataHandler)
    mock.utterances_train = ["hello", "hi", "goodbye"]
    mock.labels_train = [0, 0, 1]
    return mock


def test_vector_index_initialization():
    index = VectorIndex(embedder_device="cpu", embedder_model_name="sergeyzh/rubert-tiny-turbo")
    assert index.embedder_device == "cpu"
