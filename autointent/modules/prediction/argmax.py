from warnings import warn

import numpy as np

from .base import DataHandler, PredictionModule


class ArgmaxPredictor(PredictionModule):
    def fit(self, data_handler: DataHandler):
        if self._data_has_oos_samples(data_handler):
            warn("Your data contains out-of-scope utterances, but ArgmaxPredictor cannot detect them. Consider different predictor")

    def predict(self, scores: list[list[float]]):
        return np.argmax(scores, axis=1)
