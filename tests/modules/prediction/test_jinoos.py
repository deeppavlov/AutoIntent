from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from autointent.modules import JinoosPredictor
from autointent.modules.prediction._utils import InvalidNumClassesError, WrongClassificationError


def detect_oos(scores: npt.NDArray[Any], labels: npt.NDArray[Any], thresh: float):
    """
    `labels`: labels without oos detection
    """
    mask = np.max(scores, axis=1) < thresh
    labels[mask] = -1

    return labels


def test_predict_returns_correct_indices(multiclass_fit_data):
    predictor = JinoosPredictor()
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9, 0, 0.1], [0.8, 0, 0.2, 0.5], [0, 0.3, 0.7, 0.1]])

    # inference
    predictions = predictor.predict(scores)
    desired = detect_oos(scores, np.array([1, 0, 2]), predictor.thresh)

    np.testing.assert_array_equal(predictions, desired)


def test_fails_on_wrong_n_classes(multiclass_fit_data):
    predictor = JinoosPredictor()
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    with pytest.raises(InvalidNumClassesError):
        predictor.predict(scores)


def test_fails_on_wrong_clf_problem(multilabel_fit_data):
    predictor = JinoosPredictor()
    with pytest.raises(WrongClassificationError):
        predictor.fit(*multilabel_fit_data)
