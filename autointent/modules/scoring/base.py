from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.context.optimization_info import ScorerArtifact
from autointent.metrics import ScoringMetricFn
from autointent.modules.base import Module


class ScoringModule(Module, ABC):
    def score(self, context: Context, metric_fn: ScoringMetricFn) -> float:
        """
        Return
        ---
        - metric calculcated on test set
        - predicted scores of test set and oos utterances
        """
        self._test_scores = self.predict(context.data_handler.utterances_test)
        res = metric_fn(context.data_handler.labels_test, self._test_scores)
        self._oos_scores = None
        if context.data_handler.has_oos_samples():
            self._oos_scores = self.predict(context.data_handler.oos_utterances)
        return res

    def get_assets(self) -> ScorerArtifact:
        return ScorerArtifact(test_scores=self._test_scores, oos_scores=self._oos_scores)

    @abstractmethod
    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        pass

    def predict_topk(self, utterances: list[str], k: int = 3) -> npt.NDArray[Any]:
        scores = self.predict(utterances)
        return get_topk(scores, k)


def get_topk(scores: npt.NDArray[Any], k: int) -> npt.NDArray[Any]:
    """
    Argument
    ---
    `scores`: np.ndarray of shape (n_samples, n_classes)

    Return
    ---
    np.ndarray of shape (n_samples, k) where each row contains indexes of topk classes (from most to least probable)
    """
    # select top scores
    top_indices = np.argpartition(scores, axis=1, kth=-k)[:, -k:]
    top_scores = scores[np.arange(len(scores))[:, None], top_indices]
    # sort them
    top_indices_sorted = np.argsort(top_scores, axis=1)[:, ::-1]
    return top_indices[np.arange(len(scores))[:, None], top_indices_sorted]  # type: ignore[no-any-return]
