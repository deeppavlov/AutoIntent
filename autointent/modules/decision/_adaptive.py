"""AdaptiveDecision module for multi-label classification with adaptive thresholds."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.custom_types import FloatFromZeroToOne, ListOfGenericLabels, ListOfLabelsWithOOS, MultiLabel
from autointent.exceptions import MismatchNumClassesError
from autointent.metrics import decision_f1
from autointent.modules.base import BaseDecision
from autointent.schemas import Tag

from ._utils import apply_tags

default_search_space = np.linspace(0, 1, num=10)
logger = logging.getLogger(__name__)


class AdaptiveDecision(BaseDecision):
    """Decision for multi-label classification using adaptive thresholds.

    The AdaptiveDecision calculates optimal thresholds based on the given
    scores and labels, ensuring the best performance on multi-label data.

    Args:
        search_space: List of threshold scaling factors to search for optimal performance.
            Defaults to a range between 0 and 1

    Examples:
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

    def __init__(self, search_space: list[FloatFromZeroToOne] | None = None) -> None:
        self.search_space = search_space if search_space is not None else default_search_space

        if any(val < 0 or val > 1 for val in self.search_space):
            msg = "Unsupported items in `search_space` arg of `AdaptiveDecision` module"
            raise ValueError(msg)

    @classmethod
    def from_context(cls, context: Context, search_space: list[FloatFromZeroToOne] | None = None) -> "AdaptiveDecision":
        """Create an AdaptiveDecision instance using a Context object.

        Args:
            context: Context containing configurations and utilities
            search_space: List of threshold scaling factors, or None for default
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
        """Fit the predictor by optimizing the threshold scaling factor.

        Args:
            scores: Array of shape (n_samples, n_classes) with predicted scores
            labels: List of true multi-label targets
            tags: List of Tag objects for mutually exclusive classes, or None

        Raises:
            WrongClassificationError: If used on non-multi-label data
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
        """Predict labels for the given scores.

        Args:
            scores: Array of shape (n_samples, n_classes) with predicted scores

        Returns:
            Array of shape (n_samples, n_classes) with predicted binary labels

        Raises:
            MismatchNumClassesError: If the number of classes does not match the trained predictor
        """
        if scores.shape[1] != self._n_classes:
            raise MismatchNumClassesError
        return multilabel_predict(scores, self._r, self.tags)


def get_adapted_threshes(r: float, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Compute adaptive thresholds based on scaling factor and scores.

    Args:
        r: Scaling factor for thresholds
        scores: Array of shape (n_samples, n_classes) with predicted scores

    Returns:
        Array of thresholds for each class and sample
    """
    return r * np.max(scores, axis=1) + (1 - r) * np.min(scores, axis=1)  # type: ignore[no-any-return]


def multilabel_predict(scores: npt.NDArray[Any], r: float, tags: list[Tag] | None) -> ListOfLabelsWithOOS:
    """Predict binary labels for multi-label classification.

    Args:
        scores: Array of shape (n_samples, n_classes) with predicted scores
        r: Scaling factor for thresholds
        tags: List of Tag objects for mutually exclusive classes, or None

    Returns:
        Array of shape (n_samples, n_classes) with predicted binary labels
    """
    thresh = get_adapted_threshes(r, scores)
    res = (scores >= thresh[:, None]).astype(int)
    if tags:
        res = apply_tags(res, scores, tags)
    y_pred: list[MultiLabel] = res.tolist()
    return [lab if sum(lab) > 0 else None for lab in y_pred]


def multilabel_score(y_true: ListOfGenericLabels, y_pred: ListOfGenericLabels) -> float:
    """Calculate the weighted F1 score for multi-label classification.

    Args:
        y_true: List of true multi-label targets
        y_pred: Array of shape (n_samples, n_classes) with predicted labels

    Returns:
        Weighted F1 score
    """
    return decision_f1(y_true, y_pred)
