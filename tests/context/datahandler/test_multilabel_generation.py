from unittest.mock import Mock

import pytest

from autointent.context.data_handler import DataHandler
from autointent.context.vector_index_client import VectorIndexClient
from tests.conftest import setup_environment


@pytest.fixture
def mock_data_handler():
    mock = Mock(spec=DataHandler)
    mock.utterances_train = ["hello", "hi", "goodbye"]
    mock.labels_train = [0, 0, 1]
    return mock


@pytest.fixture
def vector_index():
    db_dir, dump_dir, logs_dir = setup_environment()
    return VectorIndexClient(device="cpu", multilabel=False, n_classes=2, db_dir=db_dir)


def test_vector_index_initialization():
    db_dir, dump_dir, logs_dir = setup_environment()
    index = VectorIndexClient(device="cpu", db_dir=db_dir)
    assert index.device == "cpu"
