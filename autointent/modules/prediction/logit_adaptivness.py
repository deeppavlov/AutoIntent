import json
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from autointent import Context

from .base import PredictionModule, get_prediction_evaluation_data

default_search_space = np.linspace(0, 1, num=10)

class LogitAdaptivnessPredictorDumpMetadata(TypedDict):
    r: list[float]


class LogitAdaptivnessPredictor(PredictionModule):
    metadata_dict_name = "metadata.json"

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

        metrics_list = []
        for r in self.search_space:
            y_pred = _detect_oos(pred_classes, best_scores, r)
            metric_value = jinoos_score(y_true, y_pred)
            metrics_list.append(metric_value)

        self._r = self.search_space[np.argmax(metrics_list)]

    def predict(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        pred_classes, best_scores = _predict(scores)
        return _detect_oos(pred_classes, best_scores, self._r)

    def dump(self, path: str) -> None:
        dump_dir = Path(path)

        metadata = LogitAdaptivnessPredictorDumpMetadata(r=self._r.tolist())

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(metadata, file, indent=4)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            metadata: LogitAdaptivnessPredictorDumpMetadata = json.load(file)

        self._r = metadata["r"]


def _find_threshes(r: float, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return r * np.max(scores, axis=0) + (1 - r) * np.min(scores, axis=0) # type: ignore[no-any-return]



def _predict(scores: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    pred_classes = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(len(scores)), pred_classes]
    return pred_classes, best_scores


def _detect_oos(classes: npt.NDArray[np.int64],
                scores: npt.NDArray[np.float64],
                r: float) -> npt.NDArray[np.int64]:
    threshes = _find_threshes(r, scores)
    rel_threshes = threshes[classes]
    classes[scores < rel_threshes] = -1  # out of scope
    return classes


def jinoos_score(y_true: npt.NDArray[np.int64], y_pred: npt.NDArray[np.int64]) -> float:
    """
    joint in and out of scope score

    ```math
    \\frac{C_{in}}{N_{in}}+\\frac{C_{oos}}{N_{oos}},
    ```

    where $C_{in}$ is the number of correctly predicted in-domain labels, and $N_{in}$ is the total number of
    in-domain labels. The same for OOS samples
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    in_domain_mask = y_true != -1
    correct_mask = y_true == y_pred

    correct_in_domain = np.sum(correct_mask & in_domain_mask)
    total_in_domain = np.sum(in_domain_mask)
    accuracy_in_domain = correct_in_domain / total_in_domain

    correct_oos = np.sum(correct_mask & ~in_domain_mask)
    total_oos = np.sum(~in_domain_mask)
    accuracy_oos = correct_oos / total_oos

    return accuracy_in_domain + accuracy_oos # type: ignore[no-any-return]
