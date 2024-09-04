from abc import abstractmethod
from typing import Callable

import numpy as np

from ..base import Context, Module


class PredictionModule(Module):
    @abstractmethod
    def fit(self, context: Context):
        pass

    @abstractmethod
    def predict(self, scores: list[list[float]]):
        pass

    def score(self, context: Context, metric_fn: Callable) -> tuple[float, np.ndarray]:
        labels, scores = get_prediction_evaluation_data(context)
        predictions = self.predict(scores)
        metric_value = metric_fn(labels, predictions)
        return metric_value, predictions

    def clear_cache(self):
        pass


def data_has_oos_samples(context: Context):
    return context.optimization_logs.get_best_oos_scores() is not None


def get_prediction_evaluation_data(context: Context):
    labels = context.data_handler.labels_test
    scores = context.optimization_logs.get_best_test_scores()

    oos_scores = context.optimization_logs.get_best_oos_scores()
    if oos_scores is not None:
        if context.multilabel:
            oos_labels = [[0] * context.n_classes] * len(oos_scores)
        else:
            oos_labels = [-1] * len(oos_scores)
        labels = np.concatenate([labels, oos_labels])
        scores = np.concatenate([scores, oos_scores])

    return labels, scores
