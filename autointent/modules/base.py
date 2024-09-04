from abc import ABC, abstractmethod
from typing import Any, Callable

from ..context import Context


class Module(ABC):
    @abstractmethod
    def fit(self, context: Context):
        pass

    @abstractmethod
    def score(self, context: Context, metric_fn: Callable) -> tuple[float, Any]:
        """
        calculates metric on test set and returns metric value
        """
        pass

    @abstractmethod
    def get_assets(self, context: Context):
        """
        return useful assets that represent intermediate data into context
        """
        pass

    @abstractmethod
    def clear_cache(self):
        """clear GPU/CPU memory"""
        pass
