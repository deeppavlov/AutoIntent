import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.context.data_handler.tags import Tag

from .base import PredictionModule, apply_tags

logger = logging.getLogger(__name__)


class ThresholdPredictor(PredictionModule):
    multilabel: bool
    tags: list[Tag]

    def __init__(self, thresh: float | list[float]) -> None:
        self.thresh = thresh

    def fit(self, context: Context) -> None:
        self.multilabel = context.multilabel
        self.tags = context.data_handler.tags

        if isinstance(self.thresh, list):
            if len(self.thresh) != context.n_classes:
                msg = "Wrong number of thresholds provided doesn't match with number of classes"
                logger.error(msg)
                raise ValueError(msg)
            self.thresh = np.array(self.thresh)

        if not context.data_handler.has_oos_samples():
            logger.warning(
                "Your data doesn't contain out-of-scope utterances."
                "Using ThresholdPredictor imposes unnecessary quality degradation."
            )

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.multilabel:
            return multilabel_predict(scores, self.thresh, self.tags)
        return multiclass_predict(scores, self.thresh)


def multiclass_predict(scores: npt.NDArray[Any], thresh: float | npt.NDArray[Any]) -> npt.NDArray[Any]:
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


def multilabel_predict(
    scores: npt.NDArray[Any], thresh: float | npt.NDArray[Any], tags: list[Tag] | None
) -> npt.NDArray[Any]:
    """
    Return
    ---
    array of binary labels, shape (n_samples, n_classes)
    """
    res = (scores >= thresh).astype(int) if isinstance(thresh, float) else (scores >= thresh[None, :]).astype(int)
    if tags:
        res = apply_tags(res, scores, tags)
    return res
