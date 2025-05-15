import numpy as np
import pytest

from autointent.exceptions import MismatchNumClassesError, WrongClassificationError
from autointent.modules.decision import ArgmaxDecision
from tests.conftest import setup_environment


def test_multiclass(multiclass_fit_data, scores):
    predictor = ArgmaxDecision()
    predictor.fit(*multiclass_fit_data)
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 2]))


def test_fails_on_wrong_n_classes(multiclass_fit_data):
    predictor = ArgmaxDecision()
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    with pytest.raises(MismatchNumClassesError):
        predictor.predict(scores)


def test_fails_on_wrong_clf_problem(multilabel_fit_data):
    predictor = ArgmaxDecision()
    with pytest.raises(WrongClassificationError):
        predictor.fit(*multilabel_fit_data)


def test_dump_load(multiclass_fit_data):
    predictor = ArgmaxDecision()
    predictor.fit(*multiclass_fit_data)
    predictions = predictor.predict(multiclass_fit_data[0])

    path = setup_environment() / "argmax_module"
    predictor.dump(path)
    del predictor

    predictor = ArgmaxDecision.load(path)
    new_predictions = predictor.predict(multiclass_fit_data[0])

    assert all(p == n for p, n in zip(predictions, new_predictions, strict=True))
