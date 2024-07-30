from typing import Callable

import numpy as np

from ..base import DataHandler, Module


class ScoringModule(Module):
    def score(self, data_handler: DataHandler, metric_fn: Callable):
        probas = self.predict(data_handler.utterances_test)
        data_handler.scores = probas   # TODO: fix the workaround
        return metric_fn(data_handler.labels_test, probas)

    def predict(self, utterances: list[str]):
        raise NotImplementedError()

    def predict_topk(self, utterances: list[str], k=3):
        """
        TODO: test this code
        """
        scores = self.predict(utterances)
        # select top scores
        top_indices = np.argpartition(scores, axis=1, kth=-k)[:, -k:]
        top_scores = scores[np.arange(len(scores))[:, None], top_indices]
        # sort them
        top_indices_sorted = np.argsort(top_scores, axis=1)[:, ::-1]
        return top_indices[np.arange(len(scores))[:, None], top_indices_sorted]
