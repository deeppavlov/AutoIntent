from warnings import warn

import numpy as np

from .base import Context, PredictionModule, apply_tags


class ThresholdPredictor(PredictionModule):
    def __init__(self, thresh: float):
        self.thresh = thresh

    def fit(self, context: Context):
        self.multilabel = context.multilabel
        self.tags = context.data_handler.tags

        if not context.data_handler.has_oos_samples():
            warn(
                "Your data doesn't contain out-of-scope utterances."
                "Using ThresholdPredictor imposes unnecessary quality degradation."
            )

    def predict(self, scores: list[list[float]]):
        if not self.multilabel:
            pred_classes = np.argmax(scores, axis=1)
            best_scores = scores[np.arange(len(scores)), pred_classes]
            pred_classes[best_scores < self.thresh] = -1  # out of scope
        else:
            pred_classes = (scores >= self.thresh).astype(int)
            if self.tags:
                pred_classes = apply_tags(pred_classes, scores, self.tags)
        return pred_classes
