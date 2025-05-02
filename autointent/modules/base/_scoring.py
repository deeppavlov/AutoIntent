"""Base class for scoring modules."""

from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt

from autointent import Context
from autointent.context.optimization_info import ScorerArtifact
from autointent.custom_types import ListOfLabels
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL
from autointent.modules.base import BaseModule


class BaseScorer(BaseModule, ABC):
    """Abstract base class for scoring modules.

    Scoring modules predict scores for utterances and evaluate their performance
    using a scoring metric.
    """

    supports_oos = False

    @abstractmethod
    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
    ) -> None:
        """Fit the scoring module to the training data.

        Args:
            utterances: List of training utterances.
            labels: List of training labels.
        """
        ...

    def score_ho(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """Evaluate the scorer on a test set and compute the specified metric.

        Args:
            context: Context containing test set and other data.
            metrics: List of metrics to compute.

        Returns:
            Computed metrics value for the test set or error code of metrics.
        """
        self.fit(*self.get_train_data(context))

        val_utterances = context.data_handler.validation_utterances(0)
        val_labels = context.data_handler.validation_labels(0)

        scores = self.predict(val_utterances)

        self._artifact = ScorerArtifact(
            train_scores=self.predict(context.data_handler.train_utterances(1)),
            validation_scores=self.predict(context.data_handler.validation_utterances(1)),
        )

        metrics_dict = SCORING_METRICS_MULTILABEL if context.is_multilabel() else SCORING_METRICS_MULTICLASS
        chosen_metrics = {name: fn for name, fn in metrics_dict.items() if name in metrics}
        return self.score_metrics_ho((val_labels, scores), chosen_metrics)

    def score_cv(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """Evaluate the scorer on a test set and compute the specified metric.

        Args:
            context: Context containing test set and other data.
            metrics: List of metrics to compute.

        Returns:
            Computed metrics value for the test set or error code of metrics.

        """
        metrics_dict = SCORING_METRICS_MULTILABEL if context.is_multilabel() else SCORING_METRICS_MULTICLASS
        chosen_metrics = {name: fn for name, fn in metrics_dict.items() if name in metrics}

        metrics_calculated, all_val_scores = self.score_metrics_cv(
            chosen_metrics, context.data_handler.validation_iterator()
        )

        self._artifact = ScorerArtifact(folded_scores=all_val_scores)

        return metrics_calculated

    def get_assets(self) -> ScorerArtifact:
        """Retrieve assets generated during scoring.

        Returns:
            ScorerArtifact containing test, validation and test scores.
        """
        return self._artifact

    def get_train_data(self, context: Context) -> tuple[list[str], ListOfLabels]:
        """Get train data.

        Args:
            context: Context to get train data from

        Returns:
            Tuple of train utterances and train labels
        """
        return context.data_handler.train_utterances(0), context.data_handler.train_labels(0)  # type: ignore[return-value]

    @abstractmethod
    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """Predict scores for a list of utterances.

        Args:
            utterances: List of utterances to score.

        Returns:
            Array of predicted scores.
        """
