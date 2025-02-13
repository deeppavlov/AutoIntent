"""Tunable predictor module."""

from typing import Any

import numpy as np
import numpy.typing as npt
import optuna
from optuna.trial import Trial

from autointent.context import Context
from autointent.custom_types import ListOfGenericLabels
from autointent.exceptions import MismatchNumClassesError
from autointent.metrics import decision_f1
from autointent.modules.abc import DecisionModule
from autointent.schemas import Tag

from ._threshold import multiclass_predict, multilabel_predict


class TunableDecision(DecisionModule):
    """
    Tunable predictor module.

    TunableDecision uses an optimization process to find the best thresholds for predicting labels
    in single-label or multi-label classification tasks. It is designed for datasets with varying
    score distributions and supports out-of-scope (OOS) detection.

    :ivar name: Name of the predictor, defaults to "tunable".
    :ivar _n_classes: Number of classes determined during fitting.
    :ivar tags: Tags for predictions, if any.

    Examples
    --------
    Single-label classification
    ===========================
    .. testcode::

        import numpy as np
        from autointent.modules import TunableDecision
        scores = np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
        labels = [1, 0, 1]
        predictor = TunableDecision(n_trials=100, seed=42)
        predictor.fit(scores, labels)
        test_scores = np.array([[0.3, 0.7], [0.5, 0.5]])
        predictions = predictor.predict(test_scores)
        print(predictions)

    .. testoutput::

        [1, 0]

    Multi-label classification
    ==========================
    .. testcode::

        labels = [[1, 0], [0, 1], [1, 1]]
        predictor = TunableDecision(n_trials=100, seed=42)
        predictor.fit(scores, labels)
        test_scores = np.array([[0.3, 0.7], [0.6, 0.4]])
        predictions = predictor.predict(test_scores)
        print(predictions)

    .. testoutput::

        [[1, 1], [1, 1]]

    """

    name = "tunable"
    _multilabel: bool
    _n_classes: int
    supports_multilabel = True
    supports_multiclass = True
    supports_oos = True
    tags: list[Tag] | None

    def __init__(
        self,
        n_trials: int = 320,
        seed: int = 0,
        tags: list[Tag] | None = None,
    ) -> None:
        """
        Initialize tunable predictor.

        :param n_trials: Number of trials
        :param seed: Seed
        :param tags: Tags
        """
        self.n_trials = n_trials
        self.seed = seed
        self.tags = tags

    @classmethod
    def from_context(cls, context: Context, n_trials: int = 320) -> "TunableDecision":
        """
        Initialize from context.

        :param context: Context
        :param n_trials: Number of trials
        """
        return cls(n_trials=n_trials, seed=context.seed, tags=context.data_handler.tags)

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: ListOfGenericLabels,
        tags: list[Tag] | None = None,
    ) -> None:
        """
        Fit module.

        When data doesn't contain out-of-scope utterances, using TunableDecision imposes unnecessary
         computational overhead.

        :param scores: Scores to fit
        :param labels: Labels to fit
        :param tags: Tags to fit
        """
        self.tags = tags
        self._validate_task(scores, labels)

        thresh_optimizer = ThreshOptimizer(
            n_classes=self._n_classes, multilabel=self._multilabel, n_trials=self.n_trials
        )

        thresh_optimizer.fit(
            probas=scores,
            labels=labels,
            seed=self.seed,
            tags=self.tags,
        )
        self.thresh = thresh_optimizer.best_thresholds

    def predict(self, scores: npt.NDArray[Any]) -> ListOfGenericLabels:
        """
        Predict the best score.

        :param scores: Scores to predict
        """
        if scores.shape[1] != self._n_classes:
            msg = "Provided scores number don't match with number of classes which predictor was trained on."
            raise MismatchNumClassesError(msg)
        if self._multilabel:
            return multilabel_predict(scores, self.thresh, self.tags)
        return multiclass_predict(scores, self.thresh)


class ThreshOptimizer:
    """Threshold optimizer."""

    def __init__(self, n_classes: int, multilabel: bool, n_trials: int | None = None) -> None:
        """
        Initialize threshold optimizer.

        :param n_classes: Number of classes
        :param multilabel: Is multilabel
        :param n_trials: Number of trials
        """
        self.n_classes = n_classes
        self.multilabel = multilabel
        self.n_trials = n_trials if n_trials is not None else n_classes * 10

    def objective(self, trial: Trial) -> float:
        """
        Objective function to optimize.

        :param trial: Trial
        """
        thresholds = np.array([trial.suggest_float(f"threshold_{i}", 0.0, 1.0) for i in range(self.n_classes)])
        if self.multilabel:
            y_pred = multilabel_predict(self.probas, thresholds, self.tags)
        else:
            y_pred = multiclass_predict(self.probas, thresholds)
        return decision_f1(self.labels, y_pred)

    def fit(
        self,
        probas: npt.NDArray[Any],
        labels: ListOfGenericLabels,
        seed: int,
        tags: list[Tag] | None = None,
    ) -> None:
        """
        Fit the optimizer.

        :param probas: Probabilities
        :param labels: Labels
        :param seed: Seed
        :param tags: Tags
        """
        self.probas = probas
        self.labels = labels
        self.tags = tags

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(self.objective, n_trials=self.n_trials)

        self.best_thresholds = np.array([study.best_params[f"threshold_{i}"] for i in range(self.n_classes)])
