import logging

import numpy as np

from .base import Context, PredictionModule, apply_tags


class ThresholdPredictor(PredictionModule):
    def __init__(self, thresh: float):
        self.thresh = thresh

    def fit(self, context: Context):
        self.multilabel = context.multilabel
        self.tags = context.data_handler.tags

        if isinstance(self.thresh, list):
            assert len(self.thresh) == context.n_classes
            self.thresh = np.array(self.thresh)

        if not context.data_handler.has_oos_samples():
            logger = logging.getLogger(__name__)
            logger.warning(
                "Your data doesn't contain out-of-scope utterances."
                "Using ThresholdPredictor imposes unnecessary quality degradation."
            )

    def predict(self, scores: list[list[float]]):
        if self.multilabel:
            return multilabel_predict(scores, self.thresh, self.tags)
        return multiclass_predict(scores, self.thresh)


def multiclass_predict(scores: list[list[float]], thresh: float | np.ndarray):
    """
    Return
    ---
    array of int labels, shape (n_samples,)
    """
    pred_classes = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(len(scores)), pred_classes]

    if isinstance(thresh, float):
        pred_classes[best_scores < thresh] = -1  # out of scope
    else:
        thresh_selected = thresh[pred_classes]
        pred_classes[best_scores < thresh_selected] = -1  # out of scope

    return pred_classes


def multilabel_predict(scores: list[list[float]], thresh: float | np.ndarray, tags):
    """
    Return
    ---
    array of binary labels, shape (n_samples, n_classes)
    """
    if isinstance(thresh, float):
        res = (scores >= thresh).astype(int)
    else:
        res = (scores >= thresh[None, :]).astype(int)
    if tags:
        res = apply_tags(res, scores, tags)
    return res
