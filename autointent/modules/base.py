from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from autointent.context import Context


class Module(ABC):
    @abstractmethod
    def fit(self, context: Context):
        pass

    @abstractmethod
    def score(self, context: Context, metric_fn: Callable) -> tuple[float, Any]:
        """
        calculates metric on test set and returns metric value
        """

    @abstractmethod
    def get_assets(self):
        """
        return useful assets that represent intermediate data into context
        """

    @abstractmethod
    def clear_cache(self):
        """clear GPU/CPU memory"""
