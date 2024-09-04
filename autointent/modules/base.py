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
        calculates metric on test set and returns metric
        and useful assets that represent intermediate data
        """
        pass

    def fit_score(self, context: Context, metric_fn: Callable) -> tuple[float, Any]:
        self.fit(context)
        return self.score(context, metric_fn)

    @abstractmethod
    def clear_cache(self):
        """clear GPU/CPU memory"""
        pass
