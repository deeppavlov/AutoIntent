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
from autointent.context.data_handler.tags import Tag
from autointent.custom_types import LABEL_TYPE, BaseMetadataDict

from .base import PredictionModule
from .threshold import multiclass_predict, multilabel_predict


class TunablePredictorDumpMetadata(BaseMetadataDict):
    multilabel: bool
    thresh: list[float]
    tags: list[Tag] | None


class TunablePredictor(PredictionModule):
    def __init__(
        self,
        n_trials: int = 10,
        seed: int = 0,
        tags: list[Tag] | None = None,
    ) -> None:
        self.n_trials = n_trials
        self.seed = seed
        self.tags = tags

    @classmethod
    def from_context(cls, context: Context, n_trials: int = 10) -> Self:
        return cls(n_trials=n_trials, seed=context.seed, tags=context.data_handler.tags)

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: list[LABEL_TYPE],
        tags: list[Tag] | None = None,
    ) -> None:
        """
        When data doesn't contain out-of-scope utterances, using
        TunablePredictor imposes unnecessary computational overhead.
        """
        self.tags = tags
        self.multilabel = isinstance(labels[0], list)
        n_classes = (
            len(labels[0]) if self.multilabel and isinstance(labels[0], list) else len(set(labels).difference([-1]))
        )

        thresh_optimizer = ThreshOptimizer(n_classes=n_classes, multilabel=self.multilabel, n_trials=self.n_trials)

        thresh_optimizer.fit(
            probas=scores,
            labels=np.array(labels),
            seed=self.seed,
            tags=self.tags,
        )
        self.thresh = thresh_optimizer.best_thresholds
        self.metadata = TunablePredictorDumpMetadata(
            multilabel=self.multilabel, thresh=self.thresh.tolist(), tags=self.tags
        )

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.multilabel:
            return multilabel_predict(scores, self.thresh, self.tags)
        return multiclass_predict(scores, self.thresh)

    def dump(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            metadata: TunablePredictorDumpMetadata = json.load(file)

        self.metadata = metadata
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
        tags: list[Tag] | None = None,
    ) -> None:
        self.probas = probas
        self.labels = labels
        self.tags = tags

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(self.objective, n_trials=self.n_trials)

        self.best_thresholds = np.array([study.best_params[f"threshold_{i}"] for i in range(self.n_classes)])
