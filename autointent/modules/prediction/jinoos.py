import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context

from .base import PredictionModule, get_prediction_evaluation_data

default_search_space = np.linspace(0, 1, num=100)


class JinoosPredictor(PredictionModule):
    def __init__(self, search_space: list[float] | None = None) -> None:
        self.search_space = search_space if search_space is not None else default_search_space

    def fit(self, context: Context) -> None:
        """
        TODO: use dev split instead of test split
        """

        if not context.data_handler.has_oos_samples():
            logger = logging.getLogger(__name__)
            logger.warning(
                "Your data doesn't contain out-of-scope utterances."
                "Using JinoosPredictor imposes unnecessary computational overhead."
            )

        y_true, scores = get_prediction_evaluation_data(context)
        pred_classes, best_scores = _predict(scores)

        metrics_list: list[float] = []
        for thresh in self.search_space:
            y_pred = _detect_oos(pred_classes, best_scores, thresh)
            metric_value = jinoos_score(y_true, y_pred)
            metrics_list.append(metric_value)

        self._thresh = self.search_space[np.argmax(metrics_list)]

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        pred_classes, best_scores = _predict(scores)
        return _detect_oos(pred_classes, best_scores, self._thresh)

    def dump(self, path: str) -> None:
        dump_dir = Path(path)

        metadata = {"thresh": self._thresh}

        with (dump_dir / "metadata.json").open("w") as file:
            json.dump(metadata, file, indent=4)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / "metadata.json").open() as file:
            metadata = json.load(file)

        self._thresh = metadata["thresh"]


def _predict(scores: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    pred_classes = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(len(scores)), pred_classes]
    return pred_classes, best_scores


def _detect_oos(classes: npt.NDArray[Any], scores: npt.NDArray[Any], thresh: float) -> npt.NDArray[Any]:
    classes[scores < thresh] = -1  # out of scope
    return classes


def jinoos_score(y_true: list[int], y_pred: list[int]) -> float:
    """
    joint in and out of scope score

    ```math
    \\frac{C_{in}}{N_{in}}+\\frac{C_{oos}}{N_{oos}},
    ```

    where $C_{in}$ is the number of correctly predicted in-domain labels, \
    and $N_{in}$ is the total number of in-domain labels. The same for OOS samples
    """
    y_true_ = np.array(y_true)
    y_pred_ = np.array(y_pred)

    in_domain_mask = y_true_ != -1
    correct_mask = y_true_ == y_pred_

    correct_in_domain = np.sum(correct_mask & in_domain_mask)
    total_in_domain = np.sum(in_domain_mask)
    accuracy_in_domain = correct_in_domain / total_in_domain

    correct_oos = np.sum(correct_mask & ~in_domain_mask)
    total_oos = np.sum(~in_domain_mask)
    accuracy_oos = correct_oos / total_oos

    return accuracy_in_domain + accuracy_oos  # type: ignore[no-any-return]
