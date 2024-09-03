from typing import Callable, Any

from ..context import Context


class Module:
    def fit(self, context: Context):
        raise NotImplementedError()

    def score(self, context: Context, metric_fn: Callable) -> tuple[float, Any]:
        """calculates metric on test set and returns metric and useful assets that represent intermediate data"""
        raise NotImplementedError()

    def fit_score(self, context: Context, metric_fn: Callable) -> tuple[float, Any]:
        self.fit(context)
        return self.score(context, metric_fn)

    def clear_cache(self):
        """clear GPU/CPU memory"""
        raise NotImplementedError()
