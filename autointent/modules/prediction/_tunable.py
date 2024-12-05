"""Tunable predictor module."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import optuna
from optuna.trial import Trial
from sklearn.metrics import f1_score
from typing_extensions import Self

from autointent.context import Context
from autointent.context.data_handler import Tag
from autointent.custom_types import BaseMetadataDict, LabelType
from autointent.modules.abc import PredictionModule

from ._threshold import multiclass_predict, multilabel_predict
from ._utils import InvalidNumClassesError


class TunablePredictorDumpMetadata(BaseMetadataDict):
    """Tunable predictor metadata."""

    multilabel: bool
    thresh: list[float]
    tags: list[Tag] | None
    n_classes: int


class TunablePredictor(PredictionModule):
    """
    Tunable predictor module.

    TunablePredictor uses an optimization process to find the best thresholds for predicting labels
    in single-label or multi-label classification tasks. It is designed for datasets with varying
    score distributions and supports out-of-scope (OOS) detection.

    :ivar name: Name of the predictor, defaults to "tunable".
    :ivar multilabel: Whether the task is multi-label classification.
    :ivar n_classes: Number of classes determined during fitting.
    :ivar tags: Tags for predictions, if any.

    Examples
    --------
    Single-label classification:
    >>> import numpy as np
    >>> from autointent.modules import TunablePredictor
    >>> scores = np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
    >>> labels = [1, 0, 1]
    >>> predictor = TunablePredictor(n_trials=100, seed=42)
    >>> predictor.fit(scores, labels)
    >>> test_scores = np.array([[0.3, 0.7], [0.5, 0.5]])
    >>> predictions = predictor.predict(test_scores)
    >>> print(predictions)
    [1 0]

    Multi-label classification:
    >>> labels = [[1, 0], [0, 1], [1, 1]]
    >>> predictor = TunablePredictor(n_trials=100, seed=42)
    >>> predictor.fit(scores, labels)
    >>> test_scores = np.array([[0.3, 0.7], [0.6, 0.4]])
    >>> predictions = predictor.predict(test_scores)
    >>> print(predictions)
    [[0 1] [1 0]]

    Saving and loading the model:
    >>> predictor.dump("outputs/")
    >>> loaded_predictor = TunablePredictor()
    >>> loaded_predictor.load("outputs/")
    >>> print(loaded_predictor.thresh)
    [0.5, 0.7]
    """

    name = "tunable"
    multilabel: bool
    n_classes: int
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
    def from_context(cls, context: Context, n_trials: int = 320) -> Self:
        """
        Initialize from context.

        :param context: Context
        :param n_trials: Number of trials
        """
        return cls(n_trials=n_trials, seed=context.seed, tags=context.data_handler.tags)

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: list[LabelType],
        tags: list[Tag] | None = None,
    ) -> None:
        """
        Fit module.

        When data doesn't contain out-of-scope utterances, using TunablePredictor imposes unnecessary
         computational overhead.

        :param scores: Scores to fit
        :param labels: Labels to fit
        :param tags: Tags to fit
        """
        self.tags = tags
        self.multilabel = isinstance(labels[0], list)
        self.n_classes = (
            len(labels[0]) if self.multilabel and isinstance(labels[0], list) else len(set(labels).difference([-1]))
        )

        thresh_optimizer = ThreshOptimizer(n_classes=self.n_classes, multilabel=self.multilabel, n_trials=self.n_trials)

        thresh_optimizer.fit(
            probas=scores,
            labels=np.array(labels),
            seed=self.seed,
            tags=self.tags,
        )
        self.thresh = thresh_optimizer.best_thresholds

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Predict the best score.

        :param scores: Scores to predict
        """
        if scores.shape[1] != self.n_classes:
            msg = "Provided scores number don't match with number of classes which predictor was trained on."
            raise InvalidNumClassesError(msg)
        if self.multilabel:
            return multilabel_predict(scores, self.thresh, self.tags)
        return multiclass_predict(scores, self.thresh)

    def dump(self, path: str) -> None:
        """
        Dump all data needed for inference.

        :param path: Path to dump
        """
        self.metadata = TunablePredictorDumpMetadata(
            multilabel=self.multilabel,
            thresh=self.thresh.tolist(),
            tags=self.tags,
            n_classes=self.n_classes,
        )

        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

    def load(self, path: str) -> None:
        """
        Load data from dump.

        :param path: Path to load
        """
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            metadata: TunablePredictorDumpMetadata = json.load(file)

        self.metadata = metadata
        self.thresh = np.array(metadata["thresh"])
        self.multilabel = metadata["multilabel"]
        self.tags = metadata["tags"]
        self.n_classes = metadata["n_classes"]


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
        return f1_score(self.labels, y_pred, average="macro")  # type: ignore[no-any-return]

    def fit(
        self,
        probas: npt.NDArray[Any],
        labels: npt.NDArray[Any],
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
