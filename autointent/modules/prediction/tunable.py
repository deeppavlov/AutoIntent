import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
import optuna
from optuna.trial import Trial
from sklearn.metrics import f1_score

from autointent import Context
from autointent.context.data_handler.tags import Tag

from .base import PredictionModule, get_prediction_evaluation_data
from .threshold import multiclass_predict, multilabel_predict


class TunablePredictorDumpMetadata(TypedDict):
    multilabel: bool
    thresh: list[float]
    tags: list[Tag]


class TunablePredictor(PredictionModule):
    metadata_dict_name: str = "metadata.json"

    def __init__(self, n_trials: int | None = None) -> None:
        self.n_trials = n_trials

    def fit(self, context: Context) -> None:
        self.tags = context.data_handler.tags
        self.multilabel = context.multilabel

        if not context.data_handler.has_oos_samples():
            logger = logging.getLogger(__name__)
            logger.warning(
                "Your data doesn't contain out-of-scope utterances."
                "Using TunablePredictor imposes unnecessary computational overhead."
            )

        thresh_optimizer = ThreshOptimizer(
            n_classes=context.n_classes, multilabel=context.multilabel, n_trials=self.n_trials
        )
        labels, scores = get_prediction_evaluation_data(context)
        thresh_optimizer.fit(
            probas=scores,
            labels=labels,
            seed=context.seed,
            tags=self.tags,
        )
        self.thresh = thresh_optimizer.best_thresholds

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.multilabel:
            return multilabel_predict(scores, self.thresh, self.tags)
        return multiclass_predict(scores, self.thresh)

    def dump(self, path: str) -> None:
        dump_dir = Path(path)

        metadata = TunablePredictorDumpMetadata(multilabel=self.multilabel, thresh=self.thresh.tolist(), tags=self.tags)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(metadata, file, indent=4)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            metadata: TunablePredictorDumpMetadata = json.load(file)

        self.thresh = np.array(metadata["thresh"])
        self.multilabel = metadata["multilabel"]
        self.tags = metadata["tags"]


class ThreshOptimizer:
    def __init__(self, n_classes: int, multilabel: bool, n_trials: int | None = None) -> None:
        self.n_classes = n_classes
        self.multilabel = multilabel
        self.n_trials = n_trials if n_trials is not None else n_classes * 10

    def objective(self, trial: Trial) -> float:
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
        tags: list[Tag],
    ) -> None:
        self.probas = probas
        self.labels = labels
        self.tags = tags

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(self.objective, n_trials=self.n_trials)

        self.best_thresholds = np.array([study.best_params[f"threshold_{i}"] for i in range(self.n_classes)])
