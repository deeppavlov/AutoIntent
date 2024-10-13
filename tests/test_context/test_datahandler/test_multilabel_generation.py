from unittest.mock import Mock

import pytest

from autointent.context.data_handler import DataHandler
from autointent.context.vector_index import (
    VectorIndex,
    _multiclass_labels_as_metadata,
    _multiclass_metadata_as_labels,
    _multilabel_labels_as_metadata,
    _multilabel_metadata_as_labels,
)


@pytest.fixture
def mock_data_handler():
    mock = Mock(spec=DataHandler)
    mock.utterances_train = ["hello", "hi", "goodbye"]
    mock.labels_train = [0, 0, 1]
    return mock


@pytest.fixture
def vector_index(tmp_path):
    return VectorIndex(device="cpu", multilabel=False, n_classes=2)


def test_vector_index_initialization():
    index = VectorIndex(device="cpu", multilabel=False, n_classes=2)
    assert index.device == "cpu"
    assert index.multilabel is False
    assert index.n_classes == 2


def test_metadata_as_labels_multiclass(vector_index):
    metadata = [{"intent_id": 0}, {"intent_id": 1}, {"intent_id": 0}]
    labels = vector_index.metadata_as_labels(metadata)
    assert labels == [0, 1, 0]


def test_metadata_as_labels_multilabel(tmp_path):
    index = VectorIndex(device="cpu", multilabel=True, n_classes=3)
    metadata = [{"0": 1, "1": 0, "2": 1}, {"0": 0, "1": 1, "2": 0}]
    labels = index.metadata_as_labels(metadata)
    assert labels == [[1, 0, 1], [0, 1, 0]]


def test_labels_as_metadata_multiclass(vector_index):
    labels = [0, 1, 0]
    metadata = vector_index.labels_as_metadata(labels)
    assert metadata == [{"intent_id": 0}, {"intent_id": 1}, {"intent_id": 0}]


def test_labels_as_metadata_multilabel():
    index = VectorIndex(device="cpu", multilabel=True, n_classes=3)
    labels = [[1, 0, 1], [0, 1, 0]]
    metadata = index.labels_as_metadata(labels)
    assert metadata == [{"0": 1, "1": 0, "2": 1}, {"0": 0, "1": 1, "2": 0}]


def test_multiclass_labels_as_metadata():
    labels = [0, 1, 2]
    metadata = _multiclass_labels_as_metadata(labels)
    assert metadata == [{"intent_id": 0}, {"intent_id": 1}, {"intent_id": 2}]


def test_multilabel_labels_as_metadata():
    labels = [[1, 0, 1], [0, 1, 0]]
    metadata = _multilabel_labels_as_metadata(labels)
    assert metadata == [{"0": 1, "1": 0, "2": 1}, {"0": 0, "1": 1, "2": 0}]


def test_multiclass_metadata_as_labels():
    metadata = [{"intent_id": 0}, {"intent_id": 1}, {"intent_id": 2}]
    labels = _multiclass_metadata_as_labels(metadata)
    assert labels == [0, 1, 2]


def test_multilabel_metadata_as_labels():
    metadata = [{"0": 1, "1": 0, "2": 1}, {"0": 0, "1": 1, "2": 0}]
    labels = _multilabel_metadata_as_labels(metadata, 3)
    assert labels == [[1, 0, 1], [0, 1, 0]]
