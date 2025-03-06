"""Argmax decision module."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.custom_types import ListOfGenericLabels
from autointent.exceptions import MismatchNumClassesError
from autointent.modules.base import BaseDecision
from autointent.schemas import Tag

logger = logging.getLogger(__name__)


class ArgmaxDecision(BaseDecision):
    """Argmax decision module.

    The ArgmaxDecision is a simple predictor that selects the class with the highest
    score (argmax) for single-label classification tasks.

    Examples:
    --------
    .. testcode::

        from autointent.modules import ArgmaxDecision
        import numpy as np
        predictor = ArgmaxDecision()
        train_scores = np.array([[0.2, 0.8], [0.7, 0.3]])
        labels = [1, 0]  # Single-label targets
        predictor.fit(train_scores, labels)
        test_scores = np.array([[0.1, 0.9], [0.6, 0.4]])
        decisions = predictor.predict(test_scores)
        print(decisions)

    .. testoutput::

        [1, 0]

    """

    name = "argmax"
    supports_oos = False
    supports_multilabel = False
    supports_multiclass = True
    _n_classes: int

    def __init__(self) -> None: ...

    @classmethod
    def from_context(cls, context: Context) -> "ArgmaxDecision":
        """Initialize from context.

        Args:
            context: Context object containing configurations and utilities
        """
        return cls()

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: ListOfGenericLabels,
        tags: list[Tag] | None = None,
    ) -> None:
        """Fit the predictor (no-op for ArgmaxDecision).

        Args:
            scores: Array of shape (n_samples, n_classes) with predicted scores
            labels: List of true labels
            tags: List of Tag objects for mutually exclusive classes, or None

        Raises:
            WrongClassificationError: If used on non-single-label data
        """
        self._validate_task(scores, labels)

    def predict(self, scores: npt.NDArray[Any]) -> list[int]:
        """Predict labels using argmax strategy.

        Args:
            scores: Array of shape (n_samples, n_classes) with predicted scores

        Returns:
            List of predicted class indices

        Raises:
            MismatchNumClassesError: If the number of classes does not match the trained predictor
        """
        if scores.shape[1] != self._n_classes:
            raise MismatchNumClassesError
        return np.argmax(scores, axis=1).tolist()  # type: ignore[no-any-return]
