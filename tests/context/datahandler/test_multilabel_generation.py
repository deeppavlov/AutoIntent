from unittest.mock import Mock

import pytest

from autointent.context.data_handler import DataHandler
from autointent.context.vector_index_client import VectorIndexClient


@pytest.fixture
def mock_data_handler():
    mock = Mock(spec=DataHandler)
    mock.utterances_train = ["hello", "hi", "goodbye"]
    mock.labels_train = [0, 0, 1]
    return mock


@pytest.fixture
def vector_index():
    return VectorIndexClient(device="cpu", multilabel=False, n_classes=2)


def test_vector_index_initialization():
    index = VectorIndexClient(device="cpu", multilabel=False, n_classes=2)
    assert index.device == "cpu"
    assert index.multilabel is False
    assert index.n_classes == 2
