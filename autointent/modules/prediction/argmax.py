from warnings import warn

import numpy as np

from .base import Context, PredictionModule, data_has_oos_samples


class ArgmaxPredictor(PredictionModule):
    def fit(self, context: Context):
        if data_has_oos_samples(context):
            warn("Your data contains out-of-scope utterances, but ArgmaxPredictor cannot detect them. Consider different predictor")

    def predict(self, scores: list[list[float]]):
        return np.argmax(scores, axis=1)
