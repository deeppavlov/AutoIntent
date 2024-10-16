import numpy as np

from autointent.modules import ThresholdPredictor


def test_predict_returns_correct_indices(context_multiclass):
    predictor = ThresholdPredictor(0.5)
    predictor.fit(context_multiclass)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))


def test_predict_returns_list(context_multiclass):
    predictor = ThresholdPredictor([0.5, 0.5, 0.5])
    predictor.fit(context_multiclass)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))


def test_predict_handles_single_class(context_multiclass):
    predictor = ThresholdPredictor(0.5)
    predictor.fit(context_multiclass)
    scores = np.array([[0.5], [0.5], [0.5]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([0, 0, 0]))
