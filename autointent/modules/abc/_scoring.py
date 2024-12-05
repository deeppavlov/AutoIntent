"""Base class for scoring modules."""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy.typing as npt

from autointent import Context
from autointent.context.optimization_info import ScorerArtifact
from autointent.custom_types import Split
from autointent.metrics import ScoringMetricFn
from autointent.modules.abc import Module


class ScoringModule(Module, ABC):
    """
    Abstract base class for scoring modules.

    Scoring modules predict scores for utterances and evaluate their performance
    using a scoring metric.
    """

    def score(
        self,
        context: Context,
        split: Literal["validation", "test"],
        metric_fn: ScoringMetricFn,
    ) -> float:
        """
        Evaluate the scorer on a test set and compute the specified metric.

        :param context: Context containing test set and other data.
        :param split: Target split
        :param metric_fn: Function to compute the scoring metric.
        :return: Computed metric value for the test set.
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

        self._oos_scores = None
        if context.data_handler.has_oos_samples():
            self._oos_scores = {
                Split.TRAIN: self.predict(context.data_handler.oos_utterances(0)),
                Split.VALIDATION: self.predict(context.data_handler.oos_utterances(1)),
                Split.TEST: self.predict(context.data_handler.oos_utterances(2)),
            }

        self._train_scores = self.predict(context.data_handler.train_utterances(1))
        self._validation_scores = self.predict(context.data_handler.validation_utterances(1))
        self._test_scores = self.predict(context.data_handler.test_utterances())

        return metric_fn(labels, scores)

    def get_assets(self) -> ScorerArtifact:
        """
        Retrieve assets generated during scoring.

        :return: ScorerArtifact containing test scores and out-of-scope (OOS) scores.
        """
        return ScorerArtifact(
            train_scores=self._train_scores,
            validation_scores=self._validation_scores,
            test_scores=self._test_scores,
            oos_scores=self._oos_scores,
        )

    @abstractmethod
    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """
        Predict scores for a list of utterances.

        :param utterances: List of utterances to score.
        :return: Array of predicted scores.
        """
