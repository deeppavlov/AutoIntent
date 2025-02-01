"""Base class for scoring modules."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.context.optimization_info import ScorerArtifact
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL
from autointent.modules.abc import Module


class ScoringModule(Module, ABC):
    """
    Abstract base class for scoring modules.

    Scoring modules predict scores for utterances and evaluate their performance
    using a scoring metric.
    """

    supports_oos = False

    def score(self, context: Context, test: bool, metrics: list[str]) -> dict[str, float]:
        """
        Evaluate the scorer on a test set and compute the specified metric.

        :param context: Context containing test set and other data.
        :param split: Target split
        :return: Computed metrics value for the test set or error code of metrics
        """
        metrics_dict = SCORING_METRICS_MULTILABEL if context.is_multilabel() else SCORING_METRICS_MULTICLASS
        chosen_metrics = {name: fn for name, fn in metrics_dict.items() if name in metrics}

        if test:
            utterances = context.data_handler.test_utterances()
            labels = context.data_handler.test_labels()
            scores = self.predict(utterances)
            return self.score_metrics((labels, scores), chosen_metrics)

        metrics_values = {name: [] for name in chosen_metrics}
        for train_utterances, train_labels, val_utterances, val_labels in context.validation_iterator(0):
            self.fit(train_utterances, train_labels)
            val_scores = self.predict(val_utterances)
            for name, fn in chosen_metrics.items():
                metrics_values[name].append(fn(val_labels, val_scores))

        return {name: np.mean(values_list) for name, values_list in metrics_values.items()}

    def get_assets(self) -> ScorerArtifact:
        """
        Retrieve assets generated during scoring.

        :return: ScorerArtifact containing test, validation and test scores.
        """
        return ScorerArtifact(
            train_scores=self._train_scores,
            validation_scores=self._validation_scores,
            test_scores=self._test_scores,
        )

    @abstractmethod
    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """
        Predict scores for a list of utterances.

        :param utterances: List of utterances to score.
        :return: Array of predicted scores.
        """
