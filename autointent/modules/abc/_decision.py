"""Predictior module."""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.context.optimization_info import DecisionArtifact
from autointent.custom_types import ListOfGenericLabels
from autointent.metrics import PREDICTION_METRICS_MULTICLASS
from autointent.modules.abc import Module
from autointent.schemas import Tag


class DecisionModule(Module, ABC):
    """Base class for decision modules."""

    @abstractmethod
    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: ListOfGenericLabels,
        tags: list[Tag] | None = None,
    ) -> None:
        """
        Fit the model.

        :param scores: Scores to fit
        :param labels: Labels to fit
        :param tags: Tags to fit
        """

    @abstractmethod
    def predict(self, scores: npt.NDArray[Any]) -> ListOfGenericLabels:
        """
        Predict the best score.

        :param scores: Scores to predict
        """

    def score(self, context: Context, split: Literal["validation", "test"], metrics: list[str]) -> dict[str, float]:
        """
        Calculate metric on test set and return metric value.

        :param context: Context to score
        :param split: Target split
        :return: Computed metrics value for the test set or error code of metrics
        """
        labels, scores = get_decision_evaluation_data(context, split)
        self._decisions = self.predict(scores)
        chosen_metrics = {name: fn for name, fn in PREDICTION_METRICS_MULTICLASS.items() if name in metrics}
        return self.score_metrics((labels, self._decisions), chosen_metrics)

    def get_assets(self) -> DecisionArtifact:
        """Return useful assets that represent intermediate data into context."""
        return DecisionArtifact(labels=self._decisions)

    def clear_cache(self) -> None:
        """Clear cache."""

    def _validate_task(self, scores: npt.NDArray[Any], labels: ListOfGenericLabels) -> None:
        self._n_classes, self._multilabel, self._oos = self._get_task_specs(labels)
        self._validate_multilabel(self._multilabel)
        self._validate_oos(self._oos, raise_error=False)
        if self._n_classes != scores.shape[1]:
            msg = (
                "There is a mismatch between provided labels and scores. "
                f"Labels contains {self._n_classes} classes, but scores contain "
                f"probabilities for {scores.shape[1]} classes."
            )
            raise ValueError(msg)


def get_decision_evaluation_data(
    context: Context,
    split: Literal["train", "validation", "test"],
) -> tuple[ListOfGenericLabels, npt.NDArray[np.float64]]:
    """
    Get decision evaluation data.

    :param context: Context
    :param split: Target split
    :return:
    """
    if split == "train":
        labels = context.data_handler.train_labels(1)
        scores = context.optimization_info.get_best_train_scores()
    elif split == "validation":
        labels = context.data_handler.validation_labels(1)
        scores = context.optimization_info.get_best_validation_scores()
    elif split == "test":
        labels = context.data_handler.test_labels()
        scores = context.optimization_info.get_best_test_scores()
    else:
        message = f"Invalid split '{split}' provided. Expected one of 'train', 'validation', or 'test'."
        raise ValueError(message)

    if scores is None:
        message = f"No '{split}' scores found in the optimization info"
        raise ValueError(message)

    return labels, scores
