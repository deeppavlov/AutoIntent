import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from autointent import Context

from .base import PredictionModule
from ...custom_types import LABEL_TYPE


class ArgmaxPredictor(PredictionModule):
    def __init__(self, has_oos_samples: bool = False) -> None:
        self.has_oos_samples = has_oos_samples

    @classmethod
    def from_context(cls, context: Context, **kwargs: Any) -> Self:
        return cls(
            has_oos_samples=context.data_handler.has_oos_samples(),
        )

    def fit(self, scores: npt.NDArray[Any], labels: list[LABEL_TYPE], *args: Any, **kwargs: dict[str, Any]) -> None:
        if self.has_oos_samples:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Your data contains out-of-scope utterances, but ArgmaxPredictor "
                "cannot detect them. Consider different predictor"
            )

    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.argmax(scores, axis=1)  # type: ignore[no-any-return]

    def load(self, path: str) -> None:
        pass

    def dump(self, path: str) -> None:
        pass
