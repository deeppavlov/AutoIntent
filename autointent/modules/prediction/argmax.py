from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from autointent import Context
from autointent.context.data_handler import Tag
from autointent.custom_types import LabelType

from .base import PredictionModule


class ArgmaxPredictor(PredictionModule):
    metadata = {}  # noqa: RUF012
    name = "argmax"

    def __init__(self) -> None:
        pass

    @classmethod
    def from_context(cls, context: Context) -> Self:
        return cls()

    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: list[LabelType],
        tags: list[Tag] | None = None,
    ) -> None:
        pass

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.argmax(scores, axis=1)  # type: ignore[no-any-return]

    def load(self, path: str) -> None:
        pass

    def dump(self, path: str) -> None:
        pass
