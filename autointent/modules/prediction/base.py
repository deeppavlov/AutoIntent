from abc import abstractmethod
from typing import Callable

import numpy as np

from ...context.data_handler import Tag
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
        self._predictions = self.predict(scores)
        metric_value = metric_fn(labels, self._predictions)
        return metric_value
    
    def get_assets(self, context: Context = None):
        return self._predictions

    def clear_cache(self):
        pass


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


def apply_tags(labels: np.ndarray, scores: np.ndarray, tags: list[Tag]):
    """
    this function is intended to be used with multilabel predictor

    If some intent classes have common tag (i.e. they are mutually exclusive) and were assigned to one sample, leave only that class that has the highest score.

    Arguments
    ---
    - `labels`: np.ndarray of shape (n_samples, n_classes) with binary labels
    - `scores`: np.ndarray of shape (n_samples, n_classes) with float values from 0..1
    - `tags`: list of Tags

    Return
    ---
    np.ndarray of shape (n_samples, n_classes) with binary labels
    """

    n_samples, _ = labels.shape
    res = np.copy(labels)

    for i in range(n_samples):
        sample_labels = labels[i].astype(bool)
        sample_scores = scores[i]

        for tag in tags:
            if any(sample_labels[idx] for idx in tag.intent_ids):
                # Find the index of the class with the highest score among the tagged indices
                max_score_index = max(tag.intent_ids, key=lambda idx: sample_scores[idx])
                # Set all other tagged indices to 0 in the res
                for idx in tag.intent_ids:
                    if idx != max_score_index:
                        res[i, idx] = 0

    return res
