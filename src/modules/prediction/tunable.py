from warnings import warn

import numpy as np

from .base import DataHandler, PredictionModule


class TunablePredictor(PredictionModule):
    def fit(self, data_handler: DataHandler):
        if not self._data_has_oos_samples(data_handler):
            warn("Your data doesn't contain out-of-scope utterances. Using JinoosPredictor imposes unnecessary computational overhead.")

        self.thresh = np.ones(data_handler.n_classes) / 2

        # TODO: optuna optimization w.r.t. some metric (maybe jinoos score?)

    def predict(self, scores: list[list[float]]):
        pred_classes = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(len(scores)), pred_classes]
        pred_classes[best_scores < self.thresh[pred_classes]] = -1     # out of scope
        return pred_classes
