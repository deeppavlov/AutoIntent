from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from autointent.context import Context
from autointent.context.optimization_info.data_models import Artifact


class Module(ABC):
    @abstractmethod
    def fit(self, context: Context) -> None:
        pass

    @abstractmethod
    def score(self, context: Context, metric_fn: Callable[[Any], Any]) -> float:
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
    def load(self, path: str) -> None:
        """load all data needed for inference"""
