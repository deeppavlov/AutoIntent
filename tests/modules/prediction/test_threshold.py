import numpy as np
import pytest

from autointent.modules import ThresholdPredictor
from autointent.modules.decision._utils import InvalidNumClassesError


@pytest.mark.parametrize(
    ("fit_fixture", "threshold", "expected"),
    [
        # Multiclass with a single scalar threshold
        ("multiclass_fit_data", 0.5, np.array([1, 0, 2])),
        # Multilabel with a single scalar threshold
        ("multilabel_fit_data", 0.5, np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])),
        # Multiclass with an array of thresholds
        ("multiclass_fit_data", np.array([0.5, 0.5, 0.8, 0.5]), np.array([1, 0, -1])),
        # Multilabel with an array of thresholds
        ("multilabel_fit_data", np.array([0.5, 0.5, 0.8, 0.5]), np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])),
    ],
)
def test_predict(fit_fixture, threshold, expected, request, scores):
    fit_data = request.getfixturevalue(fit_fixture)

    predictor = ThresholdPredictor(threshold)
    predictor.fit(*fit_data)
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, expected)


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
