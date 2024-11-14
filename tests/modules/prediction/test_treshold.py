import numpy as np

from autointent.context.data_handler import DataHandler
from autointent.modules import KNNScorer, ThresholdPredictor


def get_fit_data(db_dir, dataset):
    data_handler = DataHandler(dataset)

    knn_params = {
        "k": 3,
        "weights": "distance",
        "model_name": "sergeyzh/rubert-tiny-turbo",
        "db_dir": db_dir,
    }
    scorer = KNNScorer(**knn_params)

    scorer.fit(data_handler.train_utterances, data_handler.train_labels)
    scores = scorer.predict(data_handler.test_utterances + data_handler.oos_utterances)
    labels = data_handler.train_labels + [-1] * len(data_handler.oos_utterances)
    return scores, labels


def test_predict_returns_correct_indices(setup_environment, dataset):
    get_db_dir, dump_dir, logs_dir = setup_environment

    predictor = ThresholdPredictor(0.5)
    predictor.fit(*get_fit_data(get_db_dir(), dataset))
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))


def test_predict_returns_list(setup_environment, dataset):
    get_db_dir, dump_dir, logs_dir = setup_environment

    predictor = ThresholdPredictor(np.array([0.5, 0.5, 0.5]), n_classes=3)
    predictor.fit(*get_fit_data(get_db_dir(), dataset))
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))


def test_predict_handles_single_class(setup_environment, dataset):
    get_db_dir, dump_dir, logs_dir = setup_environment

    predictor = ThresholdPredictor(0.5)
    predictor.fit(*get_fit_data(get_db_dir(), dataset))
    scores = np.array([[0.5], [0.5], [0.5]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([0, 0, 0]))
