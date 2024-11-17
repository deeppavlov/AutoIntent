"""Argmax prediction module."""

from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from autointent import Context
from autointent.context.data_handler import Tag
from autointent.custom_types import LabelType

from .base import PredictionModule


class ArgmaxPredictor(PredictionModule):
    """Argmax prediction module."""

    metadata = {}  # noqa: RUF012
    name = "argmax"

    def __init__(self) -> None:
        """Init."""

    @classmethod
    def from_context(cls, context: Context) -> Self:
        """
        Initialize form context.

        :param context: Context
        :return:
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
        :return:
        """

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Predict the argmax.

        :param scores: Scores to predict
        :return:
        """
        return np.argmax(scores, axis=1)  # type: ignore[no-any-return]

    def load(self, path: str) -> None:
        """Load."""

    def dump(self, path: str) -> None:
        """Dump."""
