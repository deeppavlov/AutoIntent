import numpy as np

from autointent.modules import ThresholdPredictor


def test_predict_returns_correct_indices(multiclass_fit_data):
    predictor = ThresholdPredictor(0.5)
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9, 0], [0.8, 0, 0.2], [0, 0.3, 0.7]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 2]))


def test_predict_returns_list(multiclass_fit_data):
    predictor = ThresholdPredictor(np.array([0.5, 0.5, 0.8]))
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9, 0], [0.8, 0, 0.2], [0, 0.3, 0.7]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, -1]))
