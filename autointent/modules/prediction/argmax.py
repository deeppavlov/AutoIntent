import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from autointent import Context

from .base import PredictionModule


class ArgmaxPredictor(PredictionModule):
    def fit(self, context: Context) -> None:
        if context.data_handler.has_oos_samples():
            logger = logging.getLogger(__name__)
            logger.warning(
                "Your data contains out-of-scope utterances, but ArgmaxPredictor "
                "cannot detect them. Consider different predictor"
            )

    def predict(self, scores: NDArray[Any]) -> NDArray[Any]:
        return np.argmax(scores, axis=1)  # type: ignore[no-any-return]
