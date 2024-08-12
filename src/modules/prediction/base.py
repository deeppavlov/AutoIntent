from typing import Callable

import numpy as np

from ..base import DataHandler, Module


class PredictionModule(Module):
    def fit(self, data_handler: DataHandler):
        raise NotImplementedError()

    def predict(self, scores: list[list[float]]):
        raise NotImplementedError()

    def score(self, data_handler: DataHandler, metric_fn: Callable) -> tuple[float, np.ndarray]:
        labels, scores = self._get_evaluation_data(data_handler)
        predictions = self.predict(scores)
        metric_value = metric_fn(labels, predictions)
        return metric_value, predictions

    def clear_cache(self):
        pass

    def _get_evaluation_data(self, data_handler: DataHandler):
        labels = data_handler.labels_test
        scores = data_handler.get_best_test_scores()

        oos_scores = data_handler.get_best_oos_scores()
        if oos_scores is not None:
            oos_labels = [-1] * len(oos_scores)
            labels = np.concatenate([labels, oos_labels])
            scores = np.concatenate([scores, oos_scores])

        return labels, scores

    def _data_has_oos_samples(self, data_handler: DataHandler):
        return (data_handler.get_best_oos_scores() is not None)
