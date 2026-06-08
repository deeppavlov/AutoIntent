from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from autointent.exceptions import MismatchNumClassesError, WrongClassificationError
from autointent.modules.decision import JinoosDecision
from tests.conftest import setup_environment

if TYPE_CHECKING:
    import numpy.typing as npt

    from tests.modules.decision.conftest import FitData


def detect_oos(scores: npt.NDArray[Any], labels: npt.NDArray[Any], thresh: float) -> npt.NDArray[Any]:
    """
    `labels`: labels without oos detection
    """
    mask = np.max(scores, axis=1) < thresh
    labels[mask] = -1

    return labels


def test_predict_returns_correct_indices(multiclass_fit_data: FitData, scores: npt.NDArray[Any]) -> None:
    predictor = JinoosDecision()
    predictor.fit(*multiclass_fit_data)
    # inference
    predictions = predictor.predict(scores)
    desired = detect_oos(scores, np.array([1, 0, 2]), predictor._thresh)

    np.testing.assert_array_equal(predictions, desired)


def test_fails_on_wrong_n_classes(multiclass_fit_data: FitData) -> None:
    predictor = JinoosDecision()
    predictor.fit(*multiclass_fit_data)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    with pytest.raises(MismatchNumClassesError):
        predictor.predict(scores)


def test_fails_on_wrong_clf_problem(multilabel_fit_data: FitData) -> None:
    predictor = JinoosDecision()
    with pytest.raises(WrongClassificationError):
        predictor.fit(*multilabel_fit_data)


def test_dump_load(multiclass_fit_data: FitData) -> None:
    predictor = JinoosDecision()
    predictor.fit(*multiclass_fit_data)
    predictions = predictor.predict(multiclass_fit_data[0])

    path = setup_environment() / "jinoos_module"
    predictor.dump(str(path))
    del predictor

    predictor = JinoosDecision.load(str(path))

    assert hasattr(predictor, "_thresh")
    assert predictor._thresh is not None
    assert isinstance(predictor._thresh, float)

    new_predictions = predictor.predict(multiclass_fit_data[0])

    assert all(p == n for p, n in zip(predictions, new_predictions, strict=True))
