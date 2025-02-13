import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules import KNNScorer


@pytest.fixture
def multiclass_fit_data(dataset):
    data_handler = DataHandler(dataset)

    knn_params = {
        "k": 3,
        "weights": "distance",
        "embedder_config": "sergeyzh/rubert-tiny-turbo",
    }
    scorer = KNNScorer(**knn_params)

    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))
    scores = scorer.predict(data_handler.validation_utterances(1))
    labels = data_handler.validation_labels(1)
    return scores, labels


@pytest.fixture
def multilabel_fit_data(dataset):
    data_handler = DataHandler(dataset.to_multilabel())

    knn_params = {
        "k": 3,
        "weights": "distance",
        "embedder_config": "sergeyzh/rubert-tiny-turbo",
    }
    scorer = KNNScorer(**knn_params)

    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))
    scores = scorer.predict(data_handler.validation_utterances(1))
    labels = data_handler.validation_labels(1)
    return scores, labels


@pytest.fixture
def scores():
    return np.array([[0.05, 0.9, 0, 0.05], [0.8, 0, 0.1, 0.1], [0, 0.2, 0.7, 0.1]])
