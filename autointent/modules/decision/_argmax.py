"""Argmax decision module."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.custom_types import ListOfGenericLabels
from autointent.exceptions import MismatchNumClassesError
from autointent.modules.abc import BaseDecision
from autointent.schemas import Tag

logger = logging.getLogger(__name__)


class ArgmaxDecision(BaseDecision):
    """
    Argmax decision module.

    The ArgmaxDecision is a simple predictor that selects the class with the highest
    score (argmax) for single-label classification tasks.

    :ivar _n_classes: Number of classes in the dataset.

    Examples
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

    def __init__(self) -> None:
        """Init."""

    @classmethod
    def from_context(cls, context: Context) -> "ArgmaxDecision":
        """
        Initialize form context.

        :param context: Context
        """
        return cls()

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: ListOfGenericLabels,
        tags: list[Tag] | None = None,
    ) -> None:
        """
        Argmax not fitting anything.

        :param scores: Scores to fit
        :param labels: Labels to fit
        :param tags: Tags to fit
        :raises WrongClassificationError: If the classification is wrong.
        """
        self._validate_task(scores, labels)

    def predict(self, scores: npt.NDArray[Any]) -> list[int]:
        """
        Predict the argmax.

        :param scores: Scores to predict
        :raises MismatchNumClassesError: If the number of classes is invalid.
        """
        if scores.shape[1] != self._n_classes:
            raise MismatchNumClassesError
        return np.argmax(scores, axis=1).tolist()  # type: ignore[no-any-return]
