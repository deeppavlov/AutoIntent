"""Tunable predictor module."""

from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt
import optuna
from optuna.trial import Trial
from pydantic import PositiveInt

from autointent.context import Context
from autointent.custom_types import ListOfGenericLabels
from autointent.exceptions import MismatchNumClassesError
from autointent.metrics import DECISION_METRICS, DecisionMetricFn
from autointent.modules.base import BaseDecision
from autointent.schemas import Tag

from ._threshold import multiclass_predict, multilabel_predict

MetricType = Literal["decision_accuracy", "decision_f1", "decision_roc_auc", "decision_precision", "decision_recall"]


class TunableDecision(BaseDecision):
    """Tunable predictor module.

    TunableDecision uses an optimization process to find the best thresholds for predicting labels
    in single-label or multi-label classification tasks. It is designed for datasets with varying
    score distributions and supports out-of-scope (OOS) detection.

    Args:
        target_metric: Metric to optimize during threshold tuning
        n_optuna_trials: Number of optimization trials
        seed: Random seed for reproducibility
        tags: Tags for predictions (if any)

    Examples:
    --------
    Single-label classification
    ===========================
    .. testcode::

        import numpy as np
        from autointent.modules import TunableDecision
        scores = np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
        labels = [1, 0, 1]
        predictor = TunableDecision(n_optuna_trials=100, seed=42)
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
        predictor = TunableDecision(n_optuna_trials=100, seed=42)
        predictor.fit(scores, labels)
        test_scores = np.array([[0.3, 0.7], [0.6, 0.4]])
        predictions = predictor.predict(test_scores)
        print(predictions)

    .. testoutput::

        [[1, 0], [1, 0]]

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
        target_metric: MetricType = "decision_accuracy",
        n_optuna_trials: PositiveInt = 320,
        seed: int | None = 0,
        tags: list[Tag] | None = None,
    ) -> None:
        self.target_metric = target_metric
        self.n_optuna_trials = n_optuna_trials
        self.seed = seed
        self.tags = tags

        if self.n_optuna_trials < 0 or not isinstance(self.n_optuna_trials, int):
            msg = "Unsupported value for `n_optuna_trial` of `TunableDecision` module"
            raise ValueError(msg)

        if self.target_metric not in get_args(MetricType):
            msg = "Unsupported value for `target_metric` of `TunableDecision` module"
            raise TypeError(msg)

    @classmethod
    def from_context(
        cls, context: Context, target_metric: MetricType = "decision_accuracy", n_optuna_trials: PositiveInt = 320
    ) -> "TunableDecision":
        """Initialize from context.

        Args:
            context: Context containing configurations and utilities
            target_metric: Metric to optimize during threshold tuning
            n_optuna_trials: Number of optimization trials
        """
        return cls(
            target_metric=target_metric,
            n_optuna_trials=n_optuna_trials,
            seed=context.seed,
            tags=context.data_handler.tags,
        )

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: ListOfGenericLabels,
        tags: list[Tag] | None = None,
    ) -> None:
        """Fit the predictor by optimizing thresholds.

        Note: When data doesn't contain out-of-scope utterances, using TunableDecision imposes
        unnecessary computational overhead.

        Args:
            scores: Array of shape (n_samples, n_classes) with predicted scores
            labels: List of true labels
            tags: Tags for predictions (if any)
        """
        self.tags = tags
        self._validate_task(scores, labels)

        metric_fn = DECISION_METRICS[self.target_metric]

        thresh_optimizer = ThreshOptimizer(
            metric_fn, n_classes=self._n_classes, multilabel=self._multilabel, n_trials=self.n_optuna_trials
        )

        thresh_optimizer.fit(
            probas=scores,
            labels=labels,
            seed=self.seed,
            tags=self.tags,
        )
        self.thresh = thresh_optimizer.best_thresholds

    def predict(self, scores: npt.NDArray[Any]) -> ListOfGenericLabels:
        """Predict labels using optimized thresholds.

        Args:
            scores: Array of shape (n_samples, n_classes) with predicted scores

        Returns:
            Predicted labels (either single-label or multi-label)

        Raises:
            MismatchNumClassesError: If number of classes in scores doesn't match training data
        """
        if scores.shape[1] != self._n_classes:
            msg = "Provided scores number don't match with number of classes which predictor was trained on."
            raise MismatchNumClassesError(msg)
        if self._multilabel:
            return multilabel_predict(scores, self.thresh, self.tags)
        return multiclass_predict(scores, self.thresh)


class ThreshOptimizer:
    """Threshold optimizer using Optuna for hyperparameter tuning."""

    def __init__(
        self, metric_fn: DecisionMetricFn, n_classes: int, multilabel: bool, n_trials: int | None = None
    ) -> None:
        """Initialize threshold optimizer.

        Args:
            metric_fn: Metric function for optimization
            n_classes: Number of classes in the dataset
            multilabel: Whether the task is multilabel
            n_trials: Number of optimization trials (defaults to n_classes * 10)
        """
        self.metric_fn = metric_fn
        self.n_classes = n_classes
        self.multilabel = multilabel
        self.n_trials = n_trials if n_trials is not None else n_classes * 10

    def objective(self, trial: Trial) -> float:
        """Objective function to optimize.

        Args:
            trial: Optuna trial object

        Returns:
            Metric value for the current thresholds
        """
        thresholds = np.array([trial.suggest_float(f"threshold_{i}", 0.0, 1.0) for i in range(self.n_classes)])
        if self.multilabel:
            y_pred = multilabel_predict(self.probas, thresholds, self.tags)
        else:
            y_pred = multiclass_predict(self.probas, thresholds)
        return self.metric_fn(self.labels, y_pred)

    def fit(
        self,
        probas: npt.NDArray[Any],
        labels: ListOfGenericLabels,
        seed: int | None,
        tags: list[Tag] | None = None,
    ) -> None:
        """Fit the optimizer by finding optimal thresholds.

        Args:
            probas: Array of shape (n_samples, n_classes) with predicted probabilities
            labels: List of true labels
            seed: Random seed for reproducibility
            tags: Tags for predictions (if any)
        """
        self.probas = probas
        self.labels = labels
        self.tags = tags

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(self.objective, n_trials=self.n_trials)

        self.best_thresholds = np.array([study.best_params[f"threshold_{i}"] for i in range(self.n_classes)])
