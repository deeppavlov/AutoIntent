import numpy as np
import pytest

from autointent.exceptions import MismatchNumClassesError
from autointent.modules import ThresholdDecision


@pytest.mark.parametrize(
    ("fit_fixture", "threshold", "expected"),
    [
        # Multiclass with a single scalar threshold
        ("multiclass_fit_data", 0.5, [1, 0, 2]),
        # Multilabel with a single scalar threshold
        ("multilabel_fit_data", 0.5, [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]),
        # Multiclass with an array of thresholds
        ("multiclass_fit_data", [0.5, 0.5, 0.8, 0.5], [1, 0, None]),
        # Multilabel with an array of thresholds
        ("multilabel_fit_data", [0.5, 0.5, 0.8, 0.5], [[0, 1, 0, 0], [1, 0, 0, 0], None]),
    ],
)
def test_predict(fit_fixture, threshold, expected, request, scores):
    fit_data = request.getfixturevalue(fit_fixture)

    predictor = ThresholdDecision(threshold)
    predictor.fit(*fit_data)
    predictions = predictor.predict(scores)
    assert predictions == expected


def test_fails_on_wrong_n_classes_predict(multiclass_fit_data):
    predictor = ThresholdDecision(thresh=0.5)
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    with pytest.raises(MismatchNumClassesError):
        predictor.predict(scores)


def test_fails_on_wrong_n_classes_fit(multiclass_fit_data):
    predictor = ThresholdDecision(thresh=[0.5])
    with pytest.raises(MismatchNumClassesError):
        predictor.fit(*multiclass_fit_data)


@pytest.mark.parametrize("fit_fixture", ["multiclass_fit_data", "multilabel_fit_data"])
def test_dump_load(fit_fixture, request, tmp_path):
    fit_data = request.getfixturevalue(fit_fixture)
    predictor = ThresholdDecision(thresh=0.3)
    predictor.fit(*fit_data)
    predictions = predictor.predict(fit_data[0])

    predictor.dump(tmp_path)
    del predictor

    predictor = ThresholdDecision.load(tmp_path)

    assert hasattr(predictor, "thresh")
    assert predictor.thresh is not None
    assert predictor.thresh == 0.3
    assert isinstance(predictor.thresh, float)

    new_predictions = predictor.predict(fit_data[0])

    assert all(p == n for p, n in zip(predictions, new_predictions, strict=True))
