import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
from sklearn.metrics import f1_score
from typing_extensions import Self

from autointent import Context
from autointent.context.data_handler import Tag
from autointent.custom_types import LabelType
from autointent.metrics.converter import transform

from .base import PredictionModule
from .utils import InvalidNumClassesError, WrongClassificationError, apply_tags

default_search_space = np.linspace(0, 1, num=10)


class AdaptivePredictorDumpMetadata(TypedDict):
    r: float
    multilabel: bool
    tags: list[Tag] | None


class AdaptivePredictor(PredictionModule):
    metadata_dict_name = "metadata.json"
    n_classes: int
    _r: float
    tags: list[Tag] | None
    name = "adapt"

    def __init__(self, search_space: list[float] | None = None) -> None:
        self.search_space = search_space if search_space is not None else default_search_space

    @classmethod
    def from_context(cls, context: Context, search_space: list[float] | None = None) -> Self:
        return cls(
            search_space=search_space,
        )

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: list[LabelType],
        tags: list[Tag] | None = None,
    ) -> None:
        self.tags = tags
        multilabel = isinstance(labels[0], list)
        if not multilabel:
            msg = """AdaptivePredictor is not designed to perform multiclass classification,
            consider using other predictor algorithms"""
            raise WrongClassificationError(msg)
        self.n_classes = (
            len(labels[0]) if self.multilabel and isinstance(labels[0], list) else len(set(labels).difference([-1]))
        )

        metrics_list = []
        for r in self.search_space:
            y_pred = multilabel_predict(scores, r, self.tags)
            metric_value = multilabel_score(labels, y_pred)
            metrics_list.append(metric_value)

        self._r = float(self.search_space[np.argmax(metrics_list)])

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if scores.shape[1] != self.n_classes:
            msg = "Provided scores number don't match with number of classes which predictor was trained on."
            raise InvalidNumClassesError(msg)
        return multilabel_predict(scores, self._r, self.tags)

    def dump(self, path: str) -> None:
        dump_dir = Path(path)

        metadata = AdaptivePredictorDumpMetadata(r=self._r, multilabel=self.multilabel, tags=self.tags)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(metadata, file, indent=4)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            metadata: AdaptivePredictorDumpMetadata = json.load(file)

        self._r = metadata["r"]
        self.multilabel = metadata["multilabel"]
        self.tags = [Tag(**tag) for tag in metadata["tags"] if metadata["tags"] and isinstance(metadata["tags"], list)]  # type: ignore[arg-type, union-attr]
        self.metadata = metadata


def _find_threshes(r: float, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return r * np.max(scores, axis=1) + (1 - r) * np.min(scores, axis=1)  # type: ignore[no-any-return]


def multilabel_predict(scores: npt.NDArray[Any], r: float, tags: list[Tag] | None) -> npt.NDArray[Any]:
    thresh = _find_threshes(r, scores)
    res = (scores >= thresh[None, :]).astype(int)  # suspicious
    if tags:
        res = apply_tags(res, scores, tags)
    return res


def multilabel_score(y_true: list[LabelType], y_pred: npt.NDArray[Any]) -> float:
    y_true_, y_pred_ = transform(y_true, y_pred)

    return f1_score(y_pred, y_true, average="weighted")  # type: ignore[no-any-return]
