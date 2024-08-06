import numpy as np

from .base import DataHandler, PredictionModule


class ArgmaxPredictor(PredictionModule):
    def fit(self, data_handler: DataHandler):
        pass

    def predict(self, scores: list[list[float]]):
        return np.argmax(scores, axis=1)
