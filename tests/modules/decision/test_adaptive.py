from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from autointent.exceptions import MismatchNumClassesError, WrongClassificationError
from autointent.modules.decision import AdaptiveDecision

if TYPE_CHECKING:
    from pathlib import Path

    from tests.modules.decision.conftest import FitData


def test_multilabel(multilabel_fit_data: FitData) -> None:
    predictor = AdaptiveDecision()
    predictor.fit(*multilabel_fit_data)
    scores = np.array([[0.2, 0.9, 0, 0], [0.8, 0, 0.6, 0], [0, 0.4, 0.7, 0]])
    predictions = predictor.predict(scores)
    desired = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])

    np.testing.assert_array_equal(predictions, desired)


def test_fails_on_wrong_n_classes_predict(multilabel_fit_data: FitData) -> None:
    predictor = AdaptiveDecision()
    predictor.fit(*multilabel_fit_data)
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    with pytest.raises(MismatchNumClassesError):
        predictor.predict(scores)


def test_fails_on_wrong_clf_problem(multiclass_fit_data: FitData) -> None:
    predictor = AdaptiveDecision()
    with pytest.raises(WrongClassificationError):
        predictor.fit(*multiclass_fit_data)


def test_dump_load(multilabel_fit_data: FitData, tmp_path: Path) -> None:
    predictor = AdaptiveDecision()
    predictor.fit(*multilabel_fit_data)
    preds = predictor.predict(multilabel_fit_data[0])

    path = tmp_path / "adaptive_module"
    predictor.dump(str(path))
    del predictor

    predictor = AdaptiveDecision.load(str(path))

    assert hasattr(predictor, "_r")
    assert predictor._r is not None
    assert isinstance(predictor._r, float)

    new_preds = predictor.predict(multilabel_fit_data[0])

    assert all(p == n for p, n in zip(preds, new_preds, strict=True))
