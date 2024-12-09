import numpy as np
import pytest

from autointent.modules import AdaptivePredictor
from autointent.modules.prediction._utils import InvalidNumClassesError, WrongClassificationError


def test_multilabel(multilabel_fit_data):
    predictor = AdaptivePredictor()
    predictor.fit(*multilabel_fit_data)
    scores = np.array([[0.2, 0.9, 0, 0], [0.8, 0, 0.6, 0], [0, 0.4, 0.7, 0]])
    predictions = predictor.predict(scores)
    desired = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]])

    np.testing.assert_array_equal(predictions, desired)


def test_fails_on_wrong_n_classes_predict(multilabel_fit_data):
    predictor = AdaptivePredictor()
    predictor.fit(*multilabel_fit_data)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    with pytest.raises(InvalidNumClassesError):
        predictor.predict(scores)


def test_fails_on_wrong_clf_problem(multiclass_fit_data):
    predictor = AdaptivePredictor()
    with pytest.raises(WrongClassificationError):
        predictor.fit(*multiclass_fit_data)
