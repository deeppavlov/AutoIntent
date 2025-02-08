"""AdaptiveDecision module for multi-label classification with adaptive thresholds."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.custom_types import ListOfGenericLabels, ListOfLabelsWithOOS, MultiLabel
from autointent.exceptions import MismatchNumClassesError
from autointent.metrics import decision_f1
from autointent.modules.abc import DecisionModule
from autointent.schemas import Tag

from ._utils import apply_tags

default_search_space = np.linspace(0, 1, num=10)
logger = logging.getLogger(__name__)


class AdaptiveDecision(DecisionModule):
    """
    Decision for multi-label classification using adaptive thresholds.

    The AdaptiveDecision calculates optimal thresholds based on the given
    scores and labels, ensuring the best performance on multi-label data.

    :ivar _n_classes: Number of classes in the dataset.
    :ivar _r: Scaling factor for thresholds.
    :ivar tags: List of Tag objects for mutually exclusive classes.
    :ivar name: Name of the predictor, defaults to "adaptive".

    Examples
    --------
    .. testcode::

        from autointent.modules.decision import AdaptiveDecision
        import numpy as np
        scores = np.array([[0.8, 0.1, 0.4], [0.2, 0.9, 0.5]])
        labels = [[1, 0, 0], [0, 1, 0]]
        predictor = AdaptiveDecision()
        predictor.fit(scores, labels)
        decisions = predictor.predict(scores)
        print(decisions)

    .. testoutput::

        [[1, 0, 1], [0, 1, 1]]

    """

    _n_classes: int
    _r: float
    tags: list[Tag] | None
    supports_multilabel = True
    supports_multiclass = False
    supports_oos = False
    name = "adaptive"

    def __init__(self, search_space: list[float] | None = None) -> None:
        """
        Initialize the AdaptiveDecision.

        :param search_space: List of threshold scaling factors to search for optimal performance.
                             Defaults to a range between 0 and 1.
        """
        self.search_space = search_space if search_space is not None else default_search_space

    @classmethod
    def from_context(cls, context: Context, search_space: list[float] | None = None) -> "AdaptiveDecision":
        """
        Create an AdaptiveDecision instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param search_space: List of threshold scaling factors, or None for default.
        :return: Initialized AdaptiveDecision instance.
        """
        return cls(
            search_space=search_space,
        )

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: ListOfGenericLabels,
        tags: list[Tag] | None = None,
    ) -> None:
        """
        Fit the predictor by optimizing the threshold scaling factor.

        :param scores: Array of shape (n_samples, n_classes) with predicted scores.
        :param labels: List of true multi-label targets.
        :param tags: List of Tag objects for mutually exclusive classes, or None.
        :raises WrongClassificationError: If used on non-multi-label data.
        """
        self.tags = tags

        self._validate_task(scores, labels)

        metrics_list = []
        for r in self.search_space:
            y_pred = multilabel_predict(scores, r, self.tags)
            metric_value = multilabel_score(labels, y_pred)
            metrics_list.append(metric_value)

        self._r = float(self.search_space[np.argmax(metrics_list)])

    def predict(self, scores: npt.NDArray[Any]) -> ListOfLabelsWithOOS:
        """
        Predict labels for the given scores.

        :param scores: Array of shape (n_samples, n_classes) with predicted scores.
        :return: Array of shape (n_samples, n_classes) with predicted binary labels.
        :raises MismatchNumClassesError: If the number of classes does not match the trained predictor.
        """
        if scores.shape[1] != self._n_classes:
            raise MismatchNumClassesError
        return multilabel_predict(scores, self._r, self.tags)


def get_adapted_threshes(r: float, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Compute adaptive thresholds based on scaling factor and scores.

    :param r: Scaling factor for thresholds.
    :param scores: Array of shape (n_samples, n_classes) with predicted scores.
    :return: Array of thresholds for each class and sample.
    """
    return r * np.max(scores, axis=1) + (1 - r) * np.min(scores, axis=1)  # type: ignore[no-any-return]


def multilabel_predict(scores: npt.NDArray[Any], r: float, tags: list[Tag] | None) -> ListOfLabelsWithOOS:
    """
    Predict binary labels for multi-label classification.

    :param scores: Array of shape (n_samples, n_classes) with predicted scores.
    :param r: Scaling factor for thresholds.
    :param tags: List of Tag objects for mutually exclusive classes, or None.
    :return: Array of shape (n_samples, n_classes) with predicted binary labels.
    """
    thresh = get_adapted_threshes(r, scores)
    res = (scores >= thresh[:, None]).astype(int)
    if tags:
        res = apply_tags(res, scores, tags)
    y_pred: list[MultiLabel] = res.tolist()  # type: ignore[assignment]
    return [lab if sum(lab) > 0 else None for lab in y_pred]


def multilabel_score(y_true: ListOfGenericLabels, y_pred: ListOfGenericLabels) -> float:
    """
    Calculate the weighted F1 score for multi-label classification.

    :param y_true: List of true multi-label targets.
    :param y_pred: Array of shape (n_samples, n_classes) with predicted labels.
    :return: Weighted F1 score.
    """
    return decision_f1(y_true, y_pred)
