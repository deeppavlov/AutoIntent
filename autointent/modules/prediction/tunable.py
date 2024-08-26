# from warnings import warn

# import numpy as np
# import optuna
# from optuna.trial import Trial
# from sklearn.metrics import f1_score

# from .base import DataHandler, PredictionModule


# class TunablePredictor(PredictionModule):
#     """this one is intened to use for multi label classification"""
#     def fit(self, data_handler: DataHandler):
#         if not self._data_has_oos_samples(data_handler):
#             warn(
#                 "Your data doesn't contain out-of-scope utterances."
#                 "Using JinoosPredictor imposes unnecessary computational overhead."
#             )

#         thresh_optimizer = ThreshOptimizer(n_classes=data_handler.n_classes)
#         thresh_optimizer.fit(
#             probas=data_handler.get_best_test_scores(),
#             labels=to_multilabel(data_handler.labels_test),
#         )
#         self.thresh = thresh_optimizer.best_thresholds

#     def predict(self, scores: list[list[float]]):
#         return (scores > self.thresh[None, :]).astype(int)


# class ThreshOptimizer:
#     def __init__(self, n_classes: int):
#         self.n_classes = n_classes

#     def objective(self, trial: Trial):
#         thresholds = [
#             trial.suggest_float(f"threshold_{i}", 0.0, 1.0)
#             for i in range(self.n_classes)
#         ]
#         y_pred = (self.probas > thresholds).astype(int)
#         score = f1_score(self.labels, y_pred, average="macro")
#         return score

#     def fit(self, probas, labels):
#         self.probas = probas
#         self.labels = labels

#         study = optuna.create_study(
#             direction="maximize", sampler=optuna.samplers.TPESampler()
#         )
#         study.optimize(self.objective, n_trials=self.n_classes * 10)

#         self.best_thresholds = np.array(
#             [study.best_params[f"threshold_{i}"] for i in range(self.n_classes)]
#         )


# def to_multilabel(labels: list[int], n_classes: int) -> list[list[int]]:
#     """
#     convert list of labels to 2d np array with ones at corresponding places (like one-hot encoding)
#     """
#     n_samples = len(labels)
#     res = np.zeros(shape=(n_samples, n_classes))
#     res[np.arange(n_samples), np.array(labels)] = 1.0
#     return res
