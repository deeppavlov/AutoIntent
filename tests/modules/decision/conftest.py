from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from autointent.context.data_handler import DataHandler
from autointent.modules.scoring import KNNScorer

if TYPE_CHECKING:
    import numpy.typing as npt

    from autointent import Dataset
    from autointent.custom_types import ListOfGenericLabels, ListOfLabels


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
    # cast to the narrower ListOfLabels accepted by KNNScorer.fit.
    scorer.fit(
        data_handler.train_utterances(0),
        cast("ListOfLabels", data_handler.train_labels(0)),
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

    scorer.fit(
        data_handler.train_utterances(0),
        cast("ListOfLabels", data_handler.train_labels(0)),
    )
    scores = scorer.predict(data_handler.validation_utterances(1))
    labels = data_handler.validation_labels(1)
    return scores, labels


@pytest.fixture
def scores() -> npt.NDArray[np.float64]:
    return np.array([[0.05, 0.9, 0, 0.05], [0.8, 0, 0.1, 0.1], [0, 0.2, 0.7, 0.1]])
