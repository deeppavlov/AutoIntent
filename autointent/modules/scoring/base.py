from abc import abstractmethod

import numpy as np

from ...metrics import ScoringMetricFn
from ..base import Context, Module


class ScoringModule(Module):
    def score(self, context: Context, metric_fn: ScoringMetricFn) -> tuple[float, np.ndarray]:
        """
        Return
        ---
        - metric calculcated on test set
        - predicted scores of test set and oos utterances
        """
        self._test_scores = self.predict(context.data_handler.utterances_test)
        metric_value = metric_fn(context.data_handler.labels_test, self._test_scores)
        return metric_value

    def get_assets(self, context: Context):
        oos_scores = None
        if context.data_handler.has_oos_samples():
            oos_scores = self.predict(context.data_handler.oos_utterances)
        return dict(test_scores=self._test_scores, oos_scores=oos_scores)

    @abstractmethod
    def predict(self, utterances: list[str]):
        pass

    def predict_topk(self, utterances: list[str], k=3):
        scores = self.predict(utterances)
        return get_topk(scores, k)


def get_topk(scores, k):
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
    return top_indices[np.arange(len(scores))[:, None], top_indices_sorted]
