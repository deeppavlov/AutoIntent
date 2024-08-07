from typing import Callable

import numpy as np

from ..base import DataHandler, Module


class PredictionModule(Module):
    def fit(self, data_handler: DataHandler):
        raise NotImplementedError()

    def predict(self, scores: list[list[float]]):
        raise NotImplementedError()

    def score(self, data_handler: DataHandler, metric_fn: Callable) -> tuple[float, np.ndarray]:
        scores = data_handler.get_best_scores()
        predictions = self.predict(scores)
        metric_value = metric_fn(data_handler.labels_test, predictions)
        return metric_value, predictions
