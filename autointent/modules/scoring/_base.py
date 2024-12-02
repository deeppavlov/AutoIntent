"""Base class for scoring modules."""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.context.optimization_info import ScorerArtifact
from autointent.metrics import ScoringMetricFn
from autointent.modules import Module


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
        :param metric_fn: Function to compute the scoring metric.
        :return: Computed metric value for the test set.
        """
        if split == "validation":
            utterances = context.data_handler.validation_utterances(0)
            labels = context.data_handler.validation_labels(0)
        else:
            utterances = context.data_handler.test_utterances()
            labels = context.data_handler.test_labels()

        scores = self.predict(utterances)

        self._oos_scores = None
        if context.data_handler.has_oos_samples():
            self._oos_scores = self.predict(context.data_handler.oos_utterances())

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

    def predict_topk(self, utterances: list[str], k: int = 3) -> npt.NDArray[Any]:
        """
        Predict the top-k most probable classes for each utterance.

        :param utterances: List of utterances to score.
        :param k: Number of top classes to return, defaults to 3.
        :return: Array of shape (n_samples, k) with indices of the top-k classes.
        """
        scores = self.predict(utterances)
        return get_topk(scores, k)


def get_topk(scores: npt.NDArray[Any], k: int) -> npt.NDArray[Any]:
    """
    Get the indices of the top-k classes for each sample.

    :param scores: Array of shape (n_samples, n_classes) with class scores.
    :param k: Number of top classes to select.
    :return: Array of shape (n_samples, k) containing indices of the top-k classes.
    """
    # Select top scores
    top_indices = np.argpartition(scores, axis=1, kth=-k)[:, -k:]
    top_scores = scores[np.arange(len(scores))[:, None], top_indices]
    # Sort them
    top_indices_sorted = np.argsort(top_scores, axis=1)[:, ::-1]
    return top_indices[np.arange(len(scores))[:, None], top_indices_sorted]  # type: ignore[no-any-return]
