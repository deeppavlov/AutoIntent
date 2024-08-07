from typing import Callable

import numpy as np

from ..base import DataHandler, Module


class ScoringModule(Module):
    def score(self, data_handler: DataHandler, metric_fn: Callable) -> tuple[float, np.ndarray]:
        """
        Return
        ---
        - metric calculcated on test set
        - predicted scores of test set
        """
        probas = self.predict(data_handler.utterances_test)
        metric_value = metric_fn(data_handler.labels_test, probas)
        return metric_value, probas

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
