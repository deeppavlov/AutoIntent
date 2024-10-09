import numpy as np
import pytest

from autointent.modules import JinoosPredictor


@pytest.mark.xfail
def test_predict_returns_correct_indices(context):
    predictor = JinoosPredictor()
    predictor.fit(context)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))


@pytest.mark.xfail
def test_predict_handles_single_class(context):
    predictor = JinoosPredictor()
    predictor.fit(context)
    scores = np.array([[0.5], [0.5], [0.5]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([0, 0, 0]))
