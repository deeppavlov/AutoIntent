"""Threshold decision module."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.custom_types import FloatFromZeroToOne, ListOfGenericLabels, MultiLabel
from autointent.exceptions import MismatchNumClassesError
from autointent.modules.base import BaseDecision
from autointent.schemas import Tag

from ._utils import apply_tags

logger = logging.getLogger(__name__)


class ThresholdDecision(BaseDecision):
    """Threshold predictor module.

    ThresholdDecision uses a predefined threshold (or array of thresholds) to predict
    labels for single-label or multi-label classification tasks.

    Args:
        thresh: Threshold for the scores, shape (n_classes,) or float

    Examples:
    --------
    Single-label classification
    ===========================
    .. testcode::

        from autointent.modules import ThresholdDecision
        import numpy as np
        scores = np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
        labels = [1, 0, 1]
        threshold = 0.5
        predictor = ThresholdDecision(thresh=threshold)
        predictor.fit(scores, labels)
        test_scores = np.array([[0.3, 0.7], [0.5, 0.5]])
        predictions = predictor.predict(test_scores)
        print(predictions)

    .. testoutput::

        [1, 0]

    Multi-label classification
    ==========================
    .. testcode::

        labels = [[1, 0], [0, 1], [1, 1]]
        predictor = ThresholdDecision(thresh=[0.5, 0.5])
        predictor.fit(scores, labels)
        test_scores = np.array([[0.3, 0.7], [0.6, 0.4]])
        predictions = predictor.predict(test_scores)
        print(predictions)

    .. testoutput::

        [[0, 1], [1, 0]]

    """

    _multilabel: bool
    _n_classes: int
    tags: list[Tag] | None
    name = "threshold"
    supports_oos = True
    supports_multilabel = True
    supports_multiclass = True

    def __init__(
        self,
        thresh: FloatFromZeroToOne | list[FloatFromZeroToOne] = 0.5,
    ) -> None:
        val_error = False
        self.thresh = thresh if isinstance(thresh, float) else np.array(thresh)
        if isinstance(thresh, float):
            val_error = val_error or thresh < 0 or thresh > 1
        else:
            val_error = val_error or any(val < 0 or val > 1 for val in thresh)

        if val_error:
            msg = "`thresh` arg of `ThresholdDecision` must contain a float from zero to one (or list of floats)."
            raise ValueError(msg)

    @classmethod
    def from_context(
        cls, context: Context, thresh: FloatFromZeroToOne | list[FloatFromZeroToOne] = 0.5
    ) -> "ThresholdDecision":
        """Initialize from context.

        Args:
            context: Context containing configurations and utilities
            thresh: Threshold for classification
        """
        return cls(
            thresh=thresh,
        )

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: ListOfGenericLabels,
        tags: list[Tag] | None = None,
    ) -> None:
        """Fit the model.

        Args:
            scores: Array of shape (n_samples, n_classes) with predicted scores
            labels: List of true labels
            tags: List of Tag objects for mutually exclusive classes, or None

        Raises:
            MismatchNumClassesError: If number of thresholds doesn't match number of classes
        """
        self.tags = tags
        self._validate_task(scores, labels)

        if not isinstance(self.thresh, float):
            if len(self.thresh) != self._n_classes:
                msg = (
                    f"Number of thresholds provided doesn't match with number of classes."
                    f" {len(self.thresh)} != {self._n_classes}"
                )
                logger.error(msg)
                raise MismatchNumClassesError(msg)
            self.thresh = np.array(self.thresh)

    def predict(self, scores: npt.NDArray[Any]) -> ListOfGenericLabels:
        """Predict labels using thresholds.

        Args:
            scores: Array of shape (n_samples, n_classes) with predicted scores

        Returns:
            Predicted labels (either single-label or multi-label)

        Raises:
            MismatchNumClassesError: If number of classes in scores doesn't match training data
        """
        if scores.shape[1] != self._n_classes:
            msg = "Provided scores number don't match with number of classes which predictor was trained on."
            raise MismatchNumClassesError(msg)
        if self._multilabel:
            return multilabel_predict(scores, self.thresh, self.tags)
        return multiclass_predict(scores, self.thresh)


def multiclass_predict(scores: npt.NDArray[Any], thresh: float | npt.NDArray[Any]) -> ListOfGenericLabels:
    """Make predictions for multiclass classification task.

    Args:
        scores: Array of shape (n_samples, n_classes) with predicted scores
        thresh: Threshold for the scores, shape (n_classes,) or float

    Returns:
        List of predicted class indices or None for OOS samples
    """
    pred_classes: npt.NDArray[Any] = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(len(scores)), pred_classes]

    if isinstance(thresh, float):
        pred_classes[best_scores < thresh] = -1  # out of scope
    else:
        thresh_selected = thresh[pred_classes]
        pred_classes[best_scores < thresh_selected] = -1  # out of scope

    y_pred: list[int] = pred_classes.tolist()
    return [lab if lab != -1 else None for lab in y_pred]


def multilabel_predict(
    scores: npt.NDArray[Any],
    thresh: float | npt.NDArray[Any],
    tags: list[Tag] | None,
) -> ListOfGenericLabels:
    """Make predictions for multilabel classification task.

    Args:
        scores: Array of shape (n_samples, n_classes) with predicted scores
        thresh: Threshold for the scores, shape (n_classes,) or float
        tags: List of Tag objects for mutually exclusive classes, or None

    Returns:
        List of predicted multi-label targets or None for OOS samples
    """
    res = (scores >= thresh).astype(int) if isinstance(thresh, float) else (scores >= thresh[None, :]).astype(int)
    if tags:
        res = apply_tags(res, scores, tags)
    y_pred: list[MultiLabel] = res.tolist()
    return [lab if sum(lab) > 0 else None for lab in y_pred]
