"""Base class for scoring modules."""

from abc import ABC, abstractmethod
from typing import Any, Literal

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

    def score(
        self,
        context: Context,
        split: Literal["validation", "test"],
    ) -> dict[str, float | str]:
        """
        Evaluate the scorer on a test set and compute the specified metric.

        :param context: Context containing test set and other data.
        :param split: Target split
        :return: Computed metrics value for the test set or error code of metrics
        """
        if split == "validation":
            utterances = context.data_handler.validation_utterances(0)
            labels = context.data_handler.validation_labels(0)
        elif split == "test":
            utterances = context.data_handler.test_utterances()
            labels = context.data_handler.test_labels()
        else:
            message = f"Invalid split '{split}' provided. Expected one of 'validation', or 'test'."
            raise ValueError(message)

        scores = self.predict(utterances)

        self._train_scores = self.predict(context.data_handler.train_utterances(1))
        self._validation_scores = self.predict(context.data_handler.validation_utterances(1))
        self._test_scores = self.predict(context.data_handler.test_utterances())

        metrics_dict = SCORING_METRICS_MULTILABEL if context.is_multilabel() else SCORING_METRICS_MULTICLASS
        return self.score_metrics((labels, scores), metrics_dict)

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
