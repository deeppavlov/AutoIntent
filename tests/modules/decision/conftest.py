from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules.scoring import KNNScorer
from tests._helpers import is_strict_labels

if TYPE_CHECKING:
    import numpy.typing as npt

    from autointent import Dataset
    from autointent.custom_types import ListOfGenericLabels


FitData = tuple["npt.NDArray[np.float64]", "ListOfGenericLabels"]


@pytest.fixture
def multiclass_fit_data(dataset: Dataset) -> FitData:
    data_handler = DataHandler(dataset)

    scorer = KNNScorer(
        k=3,
        weights="distance",
        embedder_config={"n_features": 32},
    )

    # Labels from split 0 are guaranteed non-OOS by DataHandler invariants;
    # narrow ListOfGenericLabels -> ListOfLabels for KNNScorer.fit via TypeGuard.
    train_labels = data_handler.train_labels(0)
    assert is_strict_labels(train_labels)
    scorer.fit(
        data_handler.train_utterances(0),
        train_labels,
    )
    scores = scorer.predict(data_handler.validation_utterances(1))
    labels = data_handler.validation_labels(1)
    return scores, labels


@pytest.fixture
def multilabel_fit_data(dataset: Dataset) -> FitData:
    data_handler = DataHandler(dataset.to_multilabel())

    scorer = KNNScorer(
        k=3,
        weights="distance",
        embedder_config={"n_features": 32},
    )

    train_labels = data_handler.train_labels(0)
    assert is_strict_labels(train_labels)
    scorer.fit(
        data_handler.train_utterances(0),
        train_labels,
    )
    scores = scorer.predict(data_handler.validation_utterances(1))
    labels = data_handler.validation_labels(1)
    return scores, labels


@pytest.fixture
def scores() -> npt.NDArray[np.float64]:
    return np.array([[0.05, 0.9, 0, 0.05], [0.8, 0, 0.1, 0.1], [0, 0.2, 0.7, 0.1]])
