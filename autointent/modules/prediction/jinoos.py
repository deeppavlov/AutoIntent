import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from autointent import Context
from autointent.context.data_handler import Tag
from autointent.custom_types import BaseMetadataDict, LabelType
from autointent.metrics.converter import transform

from .base import PredictionModule
from .utils import InvalidNumClassesError, WrongClassificationError

default_search_space = np.linspace(0, 1, num=100)


class JinoosPredictorDumpMetadata(BaseMetadataDict):
    thresh: float


class JinoosPredictor(PredictionModule):
    thresh: float
    name = "jinoos"
    n_classes: int

    def __init__(
        self,
        search_space: list[float] | None = None,
    ) -> None:
        self.search_space = np.array(search_space) if search_space is not None else default_search_space

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
        """
        TODO: use dev split instead of test split
        """
        multilabel = isinstance(labels[0], list)
        if multilabel:
            msg = "JinoosPredictor is compatible with single-label classification only"
            raise WrongClassificationError(msg)
        self.n_classes = (
            len(labels[0]) if multilabel and isinstance(labels[0], list) else len(set(labels).difference([-1]))
        )

        pred_classes, best_scores = _predict(scores)

        metrics_list: list[float] = []
        for thresh in self.search_space:
            y_pred = _detect_oos(pred_classes, best_scores, thresh)
            metric_value = jinoos_score(labels, y_pred)
            metrics_list.append(metric_value)

        self.thresh = float(self.search_space[np.argmax(metrics_list)])

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if scores.shape[1] != self.n_classes:
            msg = "Provided scores number don't match with number of classes which predictor was trained on."
            raise InvalidNumClassesError(msg)
        pred_classes, best_scores = _predict(scores)
        return _detect_oos(pred_classes, best_scores, self.thresh)

    def dump(self, path: str) -> None:
        self.metadata = JinoosPredictorDumpMetadata(thresh=self.thresh)

        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            metadata: JinoosPredictorDumpMetadata = json.load(file)

        self.thresh = metadata["thresh"]
        self.metadata = metadata


def _predict(scores: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    pred_classes = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(len(scores)), pred_classes]
    return pred_classes, best_scores


def _detect_oos(classes: npt.NDArray[Any], scores: npt.NDArray[Any], thresh: float) -> npt.NDArray[Any]:
    classes[scores < thresh] = -1  # out of scope
    return classes


def jinoos_score(y_true: list[LabelType] | npt.NDArray[Any], y_pred: list[LabelType] | npt.NDArray[Any]) -> float:
    """
    joint in and out of scope score

    ```math
    \\frac{C_{in}}{N_{in}}+\\frac{C_{oos}}{N_{oos}},
    ```

    where $C_{in}$ is the number of correctly predicted in-domain labels, \
    and $N_{in}$ is the total number of in-domain labels. The same for OOS samples
    """
    y_true_, y_pred_ = transform(y_true, y_pred)

    in_domain_mask = y_true_ != -1
    correct_mask = y_true_ == y_pred_

    correct_in_domain = np.sum(correct_mask & in_domain_mask)
    total_in_domain = np.sum(in_domain_mask)
    accuracy_in_domain = correct_in_domain / total_in_domain

    correct_oos = np.sum(correct_mask & ~in_domain_mask)
    total_oos = np.sum(~in_domain_mask)
    accuracy_oos = correct_oos / total_oos

    return accuracy_in_domain + accuracy_oos  # type: ignore[no-any-return]
