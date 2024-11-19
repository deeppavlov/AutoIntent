from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.context.data_handler import Tag
from autointent.context.optimization_info import PredictorArtifact
from autointent.custom_types import LabelType
from autointent.metrics import PredictionMetricFn
from autointent.modules.base import Module


class PredictionModule(Module, ABC):
    @abstractmethod
    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: list[LabelType],
        tags: list[Tag] | None = None,
    ) -> None:
        pass

    @abstractmethod
    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        pass

    def score(self, context: Context, metric_fn: PredictionMetricFn) -> float:
        labels, scores = get_prediction_evaluation_data(context)
        self._predictions = self.predict(scores)
        return metric_fn(labels, self._predictions)

    def get_assets(self) -> PredictorArtifact:
        return PredictorArtifact(labels=self._predictions)

    def clear_cache(self) -> None:
        pass


def get_prediction_evaluation_data(
    context: Context,
) -> tuple[list[LabelType], npt.NDArray[Any]]:
    labels = np.array(context.data_handler.labels_test)
    scores = context.optimization_info.get_best_test_scores()

    if scores is None:
        msg = "No test scores found in the optimization info"
        raise ValueError(msg)

    oos_scores = context.optimization_info.get_best_oos_scores()
    return_scores = scores
    if oos_scores is not None:
        oos_labels = (
            [[0] * context.get_n_classes()] * len(oos_scores) if context.is_multilabel() else [-1] * len(oos_scores)  # type: ignore[list-item]
        )
        labels = np.concatenate([labels, np.array(oos_labels)])
        return_scores = np.concatenate([scores, oos_scores])

    return labels.tolist(), return_scores
