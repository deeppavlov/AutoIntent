from warnings import warn

import numpy as np
import optuna
from optuna.trial import Trial
from sklearn.metrics import f1_score

from .base import Context, PredictionModule, data_has_oos_samples


class TunablePredictor(PredictionModule):
    """this one is intened to use for multi label classification"""

    def fit(self, context: Context):
        if not data_has_oos_samples(context):
            warn(
                "Your data doesn't contain out-of-scope utterances."
                "Using TunablePredictor imposes unnecessary computational overhead."
            )

        thresh_optimizer = ThreshOptimizer(n_classes=context.n_classes)
        thresh_optimizer.fit(
            probas=context.optimization_logs.get_best_test_scores(),
            labels=context.data_handler.labels_test,
            seed=context.seed,
        )
        self.thresh = thresh_optimizer.best_thresholds

    def predict(self, scores: list[list[float]]):
        return (scores > self.thresh[None, :]).astype(int)


class ThreshOptimizer:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def objective(self, trial: Trial):
        thresholds = [trial.suggest_float(f"threshold_{i}", 0.0, 1.0) for i in range(self.n_classes)]
        y_pred = (self.probas > thresholds).astype(int)
        score = f1_score(self.labels, y_pred, average="macro")
        return score

    def fit(self, probas, labels, seed):
        self.probas = probas
        self.labels = labels

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(self.objective, n_trials=self.n_classes * 10)

        self.best_thresholds = np.array([study.best_params[f"threshold_{i}"] for i in range(self.n_classes)])
        print(self.best_thresholds)
