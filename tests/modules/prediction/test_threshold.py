import numpy as np
import pytest

from autointent.modules import ThresholdPredictor
from autointent.modules.prediction._utils import InvalidNumClassesError


def test_multiclass(multiclass_fit_data):
    predictor = ThresholdPredictor(0.5)
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9, 0, 0.1], [0.8, 0, 0.2, 0.5], [0, 0.3, 0.7, 0.1]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 2]))


def test_multilabel(multilabel_fit_data):
    predictor = ThresholdPredictor(thresh=0.5)
    predictor.fit(*multilabel_fit_data)
    scores = np.array([[0.1, 0.9, 0, 0.1], [0.8, 0, 0.2, 0.5], [0, 0.3, 0.7, 0.1]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([[0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]]))


def test_multiclass_list(multiclass_fit_data):
    predictor = ThresholdPredictor(np.array([0.5, 0.5, 0.8, 0.5]))
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9, 0, 0.1], [0.8, 0, 0.2, 0.5], [0, 0.3, 0.7, 0.1]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, -1]))


def test_multilabel_list(multilabel_fit_data):
    predictor = ThresholdPredictor(np.array([0.5, 0.5, 0.8, 0.5]))
    predictor.fit(*multilabel_fit_data)
    scores = np.array([[0.1, 0.9, 0, 0.1], [0.8, 0, 0.2, 0.5], [0, 0.3, 0.7, 0.1]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([[0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]]))


def test_fails_on_wrong_n_classes_predict(multiclass_fit_data):
    predictor = ThresholdPredictor(thresh=0.5)
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    with pytest.raises(InvalidNumClassesError):
        predictor.predict(scores)


def test_fails_on_wrong_n_classes_fit(multiclass_fit_data):
    predictor = ThresholdPredictor(thresh=[0.5])
    with pytest.raises(InvalidNumClassesError):
        predictor.fit(*multiclass_fit_data)
