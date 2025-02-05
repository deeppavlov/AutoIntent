"""Base class for scoring modules."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.context.optimization_info import ScorerArtifact
from autointent.custom_types import ListOfLabels
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL
from autointent.modules.abc import Module


class ScoringModule(Module, ABC):
    """
    Abstract base class for scoring modules.

    Scoring modules predict scores for utterances and evaluate their performance
    using a scoring metric.
    """

    supports_oos = False

    def score_ho(self, context: Context, metrics: list[str]) -> dict[str, float]:
        utterances = context.data_handler.validation_utterances(0)
        labels = context.data_handler.validation_labels(0)

        scores = self.predict(utterances)

        self._artifact = ScorerArtifact(
            train_scores=self.predict(context.data_handler.train_utterances(1)),
            validation_scores=self.predict(context.data_handler.validation_utterances(1)),
        )

        metrics_dict = SCORING_METRICS_MULTILABEL if context.is_multilabel() else SCORING_METRICS_MULTICLASS
        chosen_metrics = {name: fn for name, fn in metrics_dict.items() if name in metrics}
        return self.score_metrics((labels, scores), chosen_metrics)

    def score_cv(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """
        Evaluate the scorer on a test set and compute the specified metric.

        :param context: Context containing test set and other data.
        :param split: Target split
        :return: Computed metrics value for the test set or error code of metrics
        """
        metrics_dict = SCORING_METRICS_MULTILABEL if context.is_multilabel() else SCORING_METRICS_MULTICLASS
        chosen_metrics = {name: fn for name, fn in metrics_dict.items() if name in metrics}

        metrics_values = {name: [] for name in chosen_metrics}
        all_val_scores = []
        for train_utterances, train_labels, val_utterances, val_labels in context.data_handler.validation_iterator():
            self.fit(train_utterances, train_labels)
            val_scores = self.predict(val_utterances)
            for name, fn in chosen_metrics.items():
                metrics_values[name].append(fn(val_labels, val_scores))
            all_val_scores.append(val_scores)

        # save all predictions unbinded to preserve folding
        self._artifact = ScorerArtifact(folded_scores=all_val_scores)

        return {name: np.mean(values_list) for name, values_list in metrics_values.items()}

    def get_assets(self) -> ScorerArtifact:
        """
        Retrieve assets generated during scoring.

        :return: ScorerArtifact containing test, validation and test scores.
        """
        return self._artifact

    def get_train_data(self, context: Context) -> tuple[list[str], ListOfLabels]:
        return (context.data_handler.train_utterances(0), context.data_handler.train_labels(0))

    @abstractmethod
    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """
        Predict scores for a list of utterances.

        :param utterances: List of utterances to score.
        :return: Array of predicted scores.
        """
