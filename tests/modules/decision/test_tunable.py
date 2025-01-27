import numpy as np
import pytest

from autointent.exceptions import MismatchNumClassesError
from autointent.modules import TunableDecision


@pytest.mark.parametrize(
    ("fixture_name", "scores", "desired"),
    [
        (
            "multiclass_fit_data",
            np.array([[0.1, 0.9, 0, 0.5], [0.8, 0, 0.2, 0.5], [0, 0.3, 0.7, 0.5]]),
            [1, None, None],
        ),
        (
            "multilabel_fit_data",
            np.array([[0.1, 0.9, 0, 0.1], [0.8, 0, 0.1, 0.1], [0, 0.2, 0.7, 0.1]]),
            [[0, 1, 0, 0], None, None],
        ),
    ],
)
def test_predict_scenarios(request, fixture_name, scores, desired):
    # Dynamically obtain fixture data
    fit_data = request.getfixturevalue(fixture_name)

    predictor = TunableDecision()
    predictor.fit(*fit_data)
    predictions = predictor.predict(scores)

    assert predictions == desired


def test_fails_on_wrong_n_classes_predict(multiclass_fit_data):
    predictor = TunableDecision()
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    with pytest.raises(MismatchNumClassesError):
        predictor.predict(scores)
