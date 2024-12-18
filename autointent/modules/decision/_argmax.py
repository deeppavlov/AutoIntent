"""Argmax decision module."""

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


class ArgmaxDecisionDumpMetadata(BaseMetadataDict):
    """Argmax predictor metadata."""

    n_classes: int


class ArgmaxDecision(DecisionModule):
    """
    Argmax decision module.

    The ArgmaxDecision is a simple predictor that selects the class with the highest
    score (argmax) for single-label classification tasks.

    :ivar n_classes: Number of classes in the dataset.

    Examples
    --------
    .. testcode::

        from autointent.modules import ArgmaxDecision
        import numpy as np
        predictor = ArgmaxDecision()
        train_scores = np.array([[0.2, 0.8, 0.0], [0.7, 0.1, 0.2]])
        labels = [1, 0]  # Single-label targets
        predictor.fit(train_scores, labels)
        test_scores = np.array([[0.1, 0.5, 0.4], [0.6, 0.3, 0.1]])
        decisions = predictor.predict(test_scores)
        print(decisions)

    .. testoutput::

        [1 0]

    """

    name = "argmax"
    n_classes: int

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
        labels: list[LabelType],
        tags: list[Tag] | None = None,
    ) -> None:
        """
        Argmax not fitting anything.

        :param scores: Scores to fit
        :param labels: Labels to fit
        :param tags: Tags to fit
        :raises WrongClassificationError: If the classification is wrong.
        """
        multilabel = isinstance(labels[0], list)
        if multilabel:
            msg = "ArgmaxDecision is compatible with single-label classifiction only"
            raise WrongClassificationError(msg)
        self.n_classes = scores.shape[1]

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Predict the argmax.

        :param scores: Scores to predict
        :raises InvalidNumClassesError: If the number of classes is invalid.
        """
        if scores.shape[1] != self.n_classes:
            msg = "Provided scores number don't match with number of classes which predictor was trained on."
            raise InvalidNumClassesError(msg)
        return np.argmax(scores, axis=1)  # type: ignore[no-any-return]

    def dump(self, path: str) -> None:
        """
        Dump.

        :param path: Dump path.
        """
        self.metadata = ArgmaxDecisionDumpMetadata(n_classes=self.n_classes)

        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

    def load(self, path: str) -> None:
        """Load."""
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            metadata: ArgmaxDecisionDumpMetadata = json.load(file)

        self.n_classes = metadata["n_classes"]
        self.metadata = metadata
