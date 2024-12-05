import numpy as np
import pytest

from autointent.modules import TunablePredictor
from autointent.modules.prediction._utils import InvalidNumClassesError


def test_multiclass(multiclass_fit_data):
    predictor = TunablePredictor()
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9, 0, 0.5], [0.8, 0, 0.2, 0.5], [0, 0.3, 0.7, 0.5]])

    predictions = predictor.predict(scores)
    desired = np.array([1, 0, 2])

    np.testing.assert_array_equal(predictions, desired)


def test_multilabel(multilabel_fit_data):
    predictor = TunablePredictor()
    predictor.fit(*multilabel_fit_data)
    scores = np.array([[0.1, 0.9, 0, 0.1], [0.8, 0, 0.1, 0.1], [0, 0.2, 0.7, 0.1]])
    predictions = predictor.predict(scores)
    desired = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])

    np.testing.assert_array_equal(predictions, desired)


def test_fails_on_wrong_n_classes_predict(multiclass_fit_data):
    predictor = TunablePredictor()
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    with pytest.raises(InvalidNumClassesError):
        predictor.predict(scores)
