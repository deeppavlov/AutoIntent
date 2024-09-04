from warnings import warn

import numpy as np

from .base import Context, PredictionModule


class ArgmaxPredictor(PredictionModule):
    def fit(self, context: Context):
        if context.data_handler.has_oos_samples():
            warn("Your data contains out-of-scope utterances, but ArgmaxPredictor cannot detect them. Consider different predictor")

    def predict(self, scores: list[list[float]]):
        return np.argmax(scores, axis=1)
