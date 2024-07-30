from typing import Callable

from ..base import DataHandler, Module


class PredictionModule(Module):
    def fit(self, data_handler: DataHandler):
        raise NotImplementedError()

    def predict(self, scores: list[list[float]]):
        raise NotImplementedError()

    def score(self, data_handler: DataHandler, metric_fn: Callable):
        predictions = self.predict(data_handler.scores)     # TODO: fix the workaround
        return metric_fn(data_handler.labels_test, predictions)
