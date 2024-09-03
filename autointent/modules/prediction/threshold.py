from warnings import warn

import numpy as np

from .base import Context, PredictionModule, data_has_oos_samples


class ThresholdPredictor(PredictionModule):
    def __init__(self, thresh: float, multilabel: bool = False):
        self.thresh = thresh
        self.multilabel = multilabel

    def fit(self, context: Context):
        if data_has_oos_samples(context):
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
        return pred_classes
