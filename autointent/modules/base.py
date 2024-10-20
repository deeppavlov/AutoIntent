from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt

from autointent.context import Context
from autointent.context.optimization_info.data_models import Artifact
from autointent.metrics import METRIC_FN


class Module(ABC):
    @abstractmethod
    def fit(self, context: Context) -> None:
        pass

    @abstractmethod
    def score(self, context: Context, metric_fn: METRIC_FN) -> float:
        """
        calculates metric on test set and returns metric value
        """

    @abstractmethod
    def get_assets(self) -> Artifact:
        """
        return useful assets that represent intermediate data into context
        """

    @abstractmethod
    def clear_cache(self) -> None:
        """clear GPU/CPU memory"""

    @abstractmethod
    def dump(self, path: str) -> None:
        """dump all data needed for inference"""

    @abstractmethod
    def load(self, path: str) -> None:
        """load all data needed for inference"""

    @abstractmethod
    def predict(self, utterances_or_scores: list[str] | npt.NDArray[Any]) -> npt.NDArray[Any]:
        """inference"""
