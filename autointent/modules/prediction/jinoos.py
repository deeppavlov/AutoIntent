import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from autointent import Context
from autointent.context.data_handler import Tag
from autointent.custom_types import LABEL_TYPE, BaseMetadataDict
from autointent.metrics.converter import transform

from .base import PredictionModule

default_search_space = np.linspace(0, 1, num=100)


class JinoosPredictorDumpMetadata(BaseMetadataDict):
    thresh: list[float]


class JinoosPredictor(PredictionModule):
    def __init__(
        self,
        search_space: list[float] | None = None,
        thresh: float | npt.NDArray[Any] = 0.5,
    ) -> None:
        self.search_space = np.array(search_space) if search_space is not None else default_search_space
        self._thresh = np.array(thresh)

    @classmethod
    def from_context(cls, context: Context, search_space: list[float] | None = None, **kwargs: dict[str, Any]) -> Self:
        return cls(
            search_space=search_space,
        )

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: list[LABEL_TYPE],
        tags: list[Tag] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        TODO: use dev split instead of test split
        """
        pred_classes, best_scores = _predict(scores)

        metrics_list: list[float] = []
        for thresh in self.search_space:
            y_pred = _detect_oos(pred_classes, best_scores, thresh)
            metric_value = jinoos_score(labels, y_pred)
            metrics_list.append(metric_value)

        self._thresh = self.search_space[np.argmax(metrics_list)]
        self.metadata = JinoosPredictorDumpMetadata(thresh=self._thresh.tolist())

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        pred_classes, best_scores = _predict(scores)
        return _detect_oos(pred_classes, best_scores, self._thresh)

    def dump(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            metadata: JinoosPredictorDumpMetadata = json.load(file)

        self._thresh = np.array(metadata["thresh"])
        self.metadata = metadata


def _predict(scores: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    pred_classes = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(len(scores)), pred_classes]
    return pred_classes, best_scores


def _detect_oos(classes: npt.NDArray[Any], scores: npt.NDArray[Any], thresh: npt.NDArray[Any]) -> npt.NDArray[Any]:
    classes[scores < thresh] = -1  # out of scope
    return classes


def jinoos_score(y_true: list[LABEL_TYPE] | npt.NDArray[Any], y_pred: list[LABEL_TYPE] | npt.NDArray[Any]) -> float:
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
