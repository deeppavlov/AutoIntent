import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
from sklearn.metrics import f1_score

from autointent import Context
from autointent.context.data_handler.tags import Tag

from .base import PredictionModule, apply_tags, get_prediction_evaluation_data

default_search_space = np.linspace(0, 1, num=10)


class AdaptivePredictorDumpMetadata(TypedDict):
    r: float
    multilabel: bool
    tags: list[Tag]


class AdaptivePredictor(PredictionModule):
    metadata_dict_name = "metadata.json"

    def __init__(self, search_space: list[float] | None = None) -> None:
        self.search_space = search_space if search_space is not None else default_search_space

    def fit(self, context: Context) -> None:
        self.multilabel = context.multilabel
        self.tags = context.data_handler.tags

        if context.data_handler.has_oos_samples():
            logger = logging.getLogger(__name__)
            logger.warning(
                "Your data contain out-of-scope utterances." "AdaptivePredictor cannot detect out-of-scope utterances."
            )

        if not self.multilabel:
            logger = logging.getLogger(__name__)
            logger.warning(
                "AdaptivePredictor results on multiclass classification are indistinguishable from ArgmaxPredictor"
            )
            self._r = 0.0

        else:
            y_true, scores = get_prediction_evaluation_data(context)

            metrics_list = []
            for r in self.search_space:
                y_pred = multilabel_predict(scores, r, self.tags)
                metric_value = multilabel_score(y_true, y_pred)
                metrics_list.append(metric_value)

            self._r = self.search_space[np.argmax(metrics_list)]

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.multilabel:
            return multilabel_predict(scores, self._r, self.tags)
        return multiclass_predict(scores)

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
        self.tags = metadata["tags"]


def _find_threshes(r: float, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return r * np.max(scores, axis=0) + (1 - r) * np.min(scores, axis=0)  # type: ignore[no-any-return]


def multilabel_predict(scores: npt.NDArray[Any], r: float, tags: list[Tag] | None) -> npt.NDArray[Any]:
    thresh = _find_threshes(r, scores)
    res = (scores >= thresh[None, :]).astype(int)
    if tags:
        res = apply_tags(res, scores, tags)
    return res


def multiclass_predict(scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.argmax(scores, axis=1)  # type: ignore[no-any-return]


def multilabel_score(y_true: list[int | list[int]], y_pred: npt.NDArray[Any]) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return f1_score(y_pred, y_true, average="weighted")  # type: ignore[no-any-return]
