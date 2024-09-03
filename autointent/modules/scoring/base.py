from typing import Callable

import numpy as np

from ..base import Context, Module


class ScoringModule(Module):
    def score(self, context: Context, metric_fn: Callable) -> tuple[float, np.ndarray]:
        """
        Return
        ---
        - metric calculcated on test set
        - predicted scores of test set and oos utterances
        """
        assets = dict(
            test_scores=self.predict(context.data_handler.utterances_test),
            oos_scores=None if len(context.data_handler.oos_utterances) == 0 else self.predict(context.data_handler.oos_utterances)
        )

        metric_value = metric_fn(context.data_handler.labels_test, assets["test_scores"])

        return metric_value, assets

    def predict(self, utterances: list[str]):
        raise NotImplementedError()

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
