from unittest.mock import Mock

import pytest

from autointent import VectorIndex
from autointent.context.data_handler import DataHandler
from tests.conftest import setup_environment


@pytest.fixture
def mock_data_handler():
    mock = Mock(spec=DataHandler)
    mock.utterances_train = ["hello", "hi", "goodbye"]
    mock.labels_train = [0, 0, 1]
    return mock


def test_vector_index_initialization():
    db_dir, dump_dir, logs_dir = setup_environment()
    index = VectorIndex(embedder_device="cpu")
    assert index.embedder_device == "cpu"
