import numpy as np
import pytest

from autointent.modules import TunablePredictor
from autointent.modules.prediction._utils import InvalidNumClassesError


@pytest.mark.parametrize(
    ("fixture_name", "scores", "desired"),
    [
        (
            "multiclass_fit_data",
            np.array([[0.1, 0.9, 0, 0.5], [0.8, 0, 0.2, 0.5], [0, 0.3, 0.7, 0.5]]),
            np.array([1, 0, 2]),
        ),
        (
            "multilabel_fit_data",
            np.array([[0.1, 0.9, 0, 0.1], [0.8, 0, 0.1, 0.1], [0, 0.2, 0.7, 0.1]]),
            np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]),
        ),
    ],
)
def test_predict_scenarios(request, fixture_name, scores, desired):
    # Dynamically obtain fixture data
    fit_data = request.getfixturevalue(fixture_name)

    predictor = TunablePredictor()
    predictor.fit(*fit_data)
    predictions = predictor.predict(scores)

    np.testing.assert_array_equal(predictions, desired)


def test_fails_on_wrong_n_classes_predict(multiclass_fit_data):
    predictor = TunablePredictor()
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    with pytest.raises(InvalidNumClassesError):
        predictor.predict(scores)
