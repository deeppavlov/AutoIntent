"""Predictior module."""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.context.optimization_info import PredictorArtifact
from autointent.custom_types import LabelType
from autointent.metrics import DecisionMetricFn
from autointent.modules.abc import Module
from autointent.schemas import Tag


class DecisionModule(Module, ABC):
    """Base class for decision modules."""

    @abstractmethod
    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: list[LabelType],
        tags: list[Tag] | None = None,
    ) -> None:
        """
        Fit the model.

        :param scores: Scores to fit
        :param labels: Labels to fit
        :param tags: Tags to fit
        """

    @abstractmethod
    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Predict the best score.

        :param scores: Scores to predict
        """

    def score(
        self,
        context: Context,
        split: Literal["validation", "test"],
        metric_fn: DecisionMetricFn,
    ) -> float:
        """
        Calculate metric on test set and return metric value.

        :param context: Context to score
        :param split: Target split
        :param metric_fn: Metric function
        :return: Score
        """
        labels, scores = get_decision_evaluation_data(context, split)
        self._decisions = self.predict(scores)
        return metric_fn(labels, self._decisions)

    def get_assets(self) -> PredictorArtifact:
        """Return useful assets that represent intermediate data into context."""
        return PredictorArtifact(labels=self._decisions)

    def clear_cache(self) -> None:
        """Clear cache."""


def get_decision_evaluation_data(
    context: Context,
    split: Literal["train", "validation", "test"],
) -> tuple[list[LabelType], npt.NDArray[np.float64]]:
    """
    Get decision evaluation data.

    :param context: Context
    :param split: Target split
    :return:
    """
    if split == "train":
        labels = np.array(context.data_handler.train_labels(1))
        scores = context.optimization_info.get_best_train_scores()
    elif split == "validation":
        labels = np.array(context.data_handler.validation_labels(1))
        scores = context.optimization_info.get_best_validation_scores()
    elif split == "test":
        labels = np.array(context.data_handler.test_labels())
        scores = context.optimization_info.get_best_test_scores()
    else:
        message = f"Invalid split '{split}' provided. Expected one of 'train', 'validation', or 'test'."
        raise ValueError(message)

    if scores is None:
        message = f"No '{split}' scores found in the optimization info"
        raise ValueError(message)

    oos_scores = context.optimization_info.get_best_oos_scores(split)
    return_scores = scores
    if oos_scores is not None:
        oos_labels = (
            [[0] * context.get_n_classes()] * len(oos_scores) if context.is_multilabel() else [-1] * len(oos_scores)  # type: ignore[list-item]
        )
        labels = np.concatenate([labels, np.array(oos_labels)])
        return_scores = np.concatenate([scores, oos_scores])

    return labels.tolist(), return_scores  # type: ignore[return-value]
