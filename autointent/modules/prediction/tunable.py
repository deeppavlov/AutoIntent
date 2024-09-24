import logging

import numpy as np
import optuna
from optuna.trial import Trial
from sklearn.metrics import f1_score

from .base import Context, PredictionModule
from .threshold import multiclass_predict, multilabel_predict


class TunablePredictor(PredictionModule):
    """this one is intened to use for multi label classification"""

    def fit(self, context: Context):
        self.tags = context.data_handler.tags
        self.multilabel = context.multilabel

        if not context.data_handler.has_oos_samples():
            logger = logging.getLogger(__name__)
            logger.warning(
                "Your data doesn't contain out-of-scope utterances."
                "Using TunablePredictor imposes unnecessary computational overhead."
            )

        thresh_optimizer = ThreshOptimizer(n_classes=context.n_classes, multilabel=context.multilabel)
        thresh_optimizer.fit(
            probas=context.optimization_logs.get_best_test_scores(),
            labels=context.data_handler.labels_test,
            seed=context.seed,
            tags=self.tags
        )
        self.thresh = thresh_optimizer.best_thresholds

    def predict(self, scores: list[list[float]]):
        if self.multilabel:
            return multilabel_predict(scores, self.thresh, self.tags)
        return multiclass_predict(scores, self.thresh)


class ThreshOptimizer:
    def __init__(self, n_classes: int, multilabel: bool):
        self.n_classes = n_classes
        self.multilabel = multilabel

    def objective(self, trial: Trial):
        thresholds = np.array([trial.suggest_float(f"threshold_{i}", 0.0, 1.0) for i in range(self.n_classes)])
        if self.multilabel:
            y_pred = multilabel_predict(self.probas, thresholds, self.tags)
        else:
            y_pred = multiclass_predict(self.probas, thresholds)
        score = f1_score(self.labels, y_pred, average="macro")
        return score

    def fit(self, probas, labels, seed, tags):
        self.probas = probas
        self.labels = labels
        self.tags = tags

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(self.objective, n_trials=self.n_classes * 10)

        self.best_thresholds = np.array([study.best_params[f"threshold_{i}"] for i in range(self.n_classes)])
        print(self.best_thresholds)
