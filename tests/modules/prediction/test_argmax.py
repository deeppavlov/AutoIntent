import numpy as np
import pytest

from autointent.modules.prediction._argmax import ArgmaxPredictor
from autointent.modules.prediction._utils import InvalidNumClassesError, WrongClassificationError


def test_multiclass(multiclass_fit_data, scores):
    predictor = ArgmaxPredictor()
    predictor.fit(*multiclass_fit_data)
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 2]))


def test_fails_on_wrong_n_classes(multiclass_fit_data):
    predictor = ArgmaxPredictor()
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    with pytest.raises(InvalidNumClassesError):
        predictor.predict(scores)


def test_fails_on_wrong_clf_problem(multilabel_fit_data):
    predictor = ArgmaxPredictor()
    with pytest.raises(WrongClassificationError):
        predictor.fit(*multilabel_fit_data)
