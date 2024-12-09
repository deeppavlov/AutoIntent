"""Jinoos predictor module."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.custom_types import BaseMetadataDict, LabelType
from autointent.modules.abc import DecisionModule
from autointent.schemas import Tag

from ._utils import InvalidNumClassesError, WrongClassificationError

default_search_space = np.linspace(0, 1, num=100)


class JinoosPredictorDumpMetadata(BaseMetadataDict):
    """Metadata for Jinoos predictor dump."""

    thresh: float
    n_classes: int


class JinoosPredictor(DecisionModule):
    """
    Jinoos predictor module.

    JinoosPredictor predicts the best scores for single-label classification tasks
    and detects out-of-scope (OOS) samples based on a threshold.

    :ivar thresh: The optimized threshold value for OOS detection.
    :ivar name: Name of the predictor, defaults to "adaptive".
    :ivar n_classes: Number of classes determined during fitting.

    Examples
    --------
    .. testcode::

        from autointent.modules import JinoosPredictor
        import numpy as np
        scores = np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
        labels = [1, 0, 1]
        search_space = [0.3, 0.5, 0.7]
        predictor = JinoosPredictor(search_space=search_space)
        predictor.fit(scores, labels)
        test_scores = np.array([[0.3, 0.7], [0.5, 0.5]])
        predictions = predictor.predict(test_scores)
        print(predictions)

    .. testoutput::

        [1 0]

    """

    thresh: float
    name = "jinoos"
    n_classes: int

    def __init__(
        self,
        search_space: list[float] | None = None,
    ) -> None:
        """
        Initialize Jinoos predictor.

        :param search_space: Search space for threshold
        """
        self.search_space = np.array(search_space) if search_space is not None else default_search_space

    @classmethod
    def from_context(cls, context: Context, search_space: list[float] | None = None) -> "JinoosPredictor":
        """
        Initialize from context.

        :param context: Context
        :param search_space: Search space
        """
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
        Fit the model.

        :param scores: Scores to fit
        :param labels: Labels to fit
        :param tags: Tags to fit
        """
        # TODO: use dev split instead of test split.
        multilabel = isinstance(labels[0], list)
        if multilabel:
            msg = "JinoosPredictor is compatible with single-label classification only"
            raise WrongClassificationError(msg)
        self.n_classes = len(set(labels).difference([-1]))

        pred_classes, best_scores = _predict(scores)

        metrics_list: list[float] = []
        for thresh in self.search_space:
            y_pred = _detect_oos(pred_classes, best_scores, thresh)
            metric_value = self.jinoos_score(labels, y_pred)
            metrics_list.append(metric_value)

        self.thresh = float(self.search_space[np.argmax(metrics_list)])

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Predict the best score.

        :param scores: Scores to predict
        """
        if scores.shape[1] != self.n_classes:
            msg = "Provided scores number don't match with number of classes which predictor was trained on."
            raise InvalidNumClassesError(msg)
        pred_classes, best_scores = _predict(scores)
        return _detect_oos(pred_classes, best_scores, self.thresh)

    def dump(self, path: str) -> None:
        """
        Dump all data needed for inference.

        :param path: Path to dump
        """
        self.metadata = JinoosPredictorDumpMetadata(thresh=self.thresh, n_classes=self.n_classes)

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
            metadata: JinoosPredictorDumpMetadata = json.load(file)

        self.thresh = metadata["thresh"]
        self.metadata = metadata
        self.n_classes = metadata["n_classes"]

    @staticmethod
    def jinoos_score(y_true: list[LabelType] | npt.NDArray[Any], y_pred: list[LabelType] | npt.NDArray[Any]) -> float:
        r"""
        Calculate Jinoos score.

        .. math::

            \\frac{C_{in}}{N_{in}}+\\frac{C_{oos}}{N_{oos}}

        where $C_{in}$ is the number of correctly predicted in-domain labels
         and $N_{in}$ is the total number of in-domain labels. The same for OOS samples

        :param y_true: True labels
        :param y_pred: Predicted labels
        """
        y_true_, y_pred_ = np.array(y_true), np.array(y_pred)

        in_domain_mask = y_true_ != -1
        correct_mask = y_true_ == y_pred_

        correct_in_domain = np.sum(correct_mask & in_domain_mask)
        total_in_domain = np.sum(in_domain_mask)
        accuracy_in_domain = correct_in_domain / total_in_domain

        correct_oos = np.sum(correct_mask & ~in_domain_mask)
        total_oos = np.sum(~in_domain_mask)
        accuracy_oos = correct_oos / total_oos

        return accuracy_in_domain + accuracy_oos  # type: ignore[no-any-return]


def _predict(scores: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """
    Predict the best score.

    :param scores: Scores to predict
    :return:
    """
    pred_classes = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(len(scores)), pred_classes]
    return pred_classes, best_scores


def _detect_oos(classes: npt.NDArray[Any], scores: npt.NDArray[Any], thresh: float) -> npt.NDArray[Any]:
    """
    Detect out of scope samples.

    :param classes: Classes to detect
    :param scores: Scores to detect
    :param thresh: Threshold to detect
    :return:
    """
    classes[scores < thresh] = -1  # out of scope
    return classes
