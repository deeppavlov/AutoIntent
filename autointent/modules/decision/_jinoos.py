"""Jinoos predictor module."""

from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.custom_types import FloatFromZeroToOne, ListOfGenericLabels
from autointent.exceptions import MismatchNumClassesError
from autointent.modules.base import BaseDecision
from autointent.schemas import Tag

default_search_space = np.linspace(0, 1, num=100)


class JinoosDecision(BaseDecision):
    """Jinoos predictor module.

    JinoosDecision predicts the best scores for single-label classification tasks
    and detects out-of-scope (OOS) samples based on a threshold.

    Args:
        search_space: List of threshold values to search through for OOS detection

    Examples:
    --------
    .. testcode::

        from autointent.modules import JinoosDecision
        import numpy as np
        scores = np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
        labels = [1, 0, 1]
        search_space = [0.3, 0.5, 0.7]
        predictor = JinoosDecision(search_space=search_space)
        predictor.fit(scores, labels)
        test_scores = np.array([[0.3, 0.7], [0.5, 0.5]])
        predictions = predictor.predict(test_scores)
        print(predictions)

    .. testoutput::

        [1, 0]

    """

    thresh: float
    name = "jinoos"
    _n_classes: int
    supports_multilabel = False
    supports_multiclass = True
    supports_oos = True

    def __init__(
        self,
        search_space: list[FloatFromZeroToOne] | None = None,
    ) -> None:
        self.search_space = np.array(search_space) if search_space is not None else default_search_space

        if any(val < 0 or val > 1 for val in self.search_space):
            msg = "Items pf `search_space` of `AdaptiveDecision` module must be a floats from zero to one"
            raise ValueError(msg)

    @classmethod
    def from_context(cls, context: Context, search_space: list[FloatFromZeroToOne] | None = None) -> "JinoosDecision":
        """Initialize from context.

        Args:
            context: Context containing configurations and utilities
            search_space: List of threshold values to search through
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
        """Fit the model.

        Args:
            scores: Array of shape (n_samples, n_classes) with predicted scores
            labels: List of true labels
            tags: List of Tag objects for mutually exclusive classes, or None
        """
        self._validate_task(scores, labels)

        pred_classes, best_scores = _predict(scores)

        metrics_list: list[float] = []
        for thresh in self.search_space:
            y_pred = _detect_oos(pred_classes, best_scores, thresh)
            metric_value = self.jinoos_score(labels, y_pred)
            metrics_list.append(metric_value)

        self._thresh = float(self.search_space[np.argmax(metrics_list)])

    def predict(self, scores: npt.NDArray[Any]) -> list[int | None]:
        """Predict the best score.

        Args:
            scores: Array of shape (n_samples, n_classes) with predicted scores

        Returns:
            List of predicted class indices or None for OOS samples

        Raises:
            MismatchNumClassesError: If the number of classes does not match the trained predictor
        """
        if scores.shape[1] != self._n_classes:
            raise MismatchNumClassesError
        pred_classes, best_scores = _predict(scores)
        y_pred: list[int] = _detect_oos(pred_classes, best_scores, self._thresh).tolist()
        return [lab if lab != -1 else None for lab in y_pred]

    @staticmethod
    def jinoos_score(y_true: ListOfGenericLabels, y_pred: npt.NDArray[Any]) -> float:
        r"""Calculate Jinoos score.

        The score is calculated as:

        .. math::

            \frac{C_{in}}{N_{in}}+\frac{C_{oos}}{N_{oos}}

        where :math:`C_{in}` is the number of correctly predicted in-domain labels
        and :math:`N_{in}` is the total number of in-domain labels. The same for OOS samples.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Combined accuracy score for in-domain and OOS samples
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
    """Predict the best score.

    Args:
        scores: Array of shape (n_samples, n_classes) with predicted scores

    Returns:
        Tuple containing:
            - Array of predicted class indices
            - Array of highest scores for each sample
    """
    pred_classes = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(len(scores)), pred_classes]
    return pred_classes, best_scores


def _detect_oos(classes: npt.NDArray[Any], scores: npt.NDArray[Any], thresh: float) -> npt.NDArray[Any]:
    """Detect out of scope samples.

    Args:
        classes: Array of predicted class indices
        scores: Array of confidence scores
        thresh: Threshold for OOS detection

    Returns:
        Array of predicted class indices with OOS samples marked as -1
    """
    classes[scores < thresh] = -1  # out of scope
    return classes
