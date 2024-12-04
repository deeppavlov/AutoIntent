import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules import KNNScorer
from tests.conftest import setup_environment


@pytest.fixture
def multiclass_fit_data(dataset):
    db_dir, dump_dir, logs_dir = setup_environment()

    data_handler = DataHandler(dataset)

    knn_params = {
        "k": 3,
        "weights": "distance",
        "embedder_name": "sergeyzh/rubert-tiny-turbo",
        "db_dir": db_dir,
    }
    scorer = KNNScorer(**knn_params)

    scorer.fit(data_handler.train_utterances(1), data_handler.train_labels(1))
    scores = scorer.predict(data_handler.validation_utterances(1) + data_handler.oos_utterances(1))
    labels = data_handler.validation_labels(1) + [-1] * len(data_handler.oos_utterances(1))
    return scores, labels


@pytest.fixture
def multilabel_fit_data(dataset):
    db_dir, dump_dir, logs_dir = setup_environment()

    data_handler = DataHandler(dataset, force_multilabel=True)

    knn_params = {
        "k": 3,
        "weights": "distance",
        "embedder_name": "sergeyzh/rubert-tiny-turbo",
        "db_dir": db_dir,
    }
    scorer = KNNScorer(**knn_params)

    scorer.fit(data_handler.train_utterances(1), data_handler.train_labels(1))
    scores = scorer.predict(data_handler.validation_utterances(1) + data_handler.oos_utterances(1))
    labels = data_handler.validation_labels(1) + [[0] * data_handler.n_classes] * len(data_handler.oos_utterances(1))
    return scores, labels
