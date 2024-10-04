import logging

import numpy as np

from .base import Context, PredictionModule


class ArgmaxPredictor(PredictionModule):
    def fit(self, context: Context):
        if context.data_handler.has_oos_samples():
            logger = logging.getLogger(__name__)
            logger.warning(
                "Your data contains out-of-scope utterances, but ArgmaxPredictor "
                "cannot detect them. Consider different predictor"
            )

    def predict(self, scores: list[list[float]]) -> list[int]:
        return np.argmax(scores, axis=1)
