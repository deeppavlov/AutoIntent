from typing import Callable, Any

from ..data_handler import DataHandler


class Module:
    def fit(self, data_handler: DataHandler):
        raise NotImplementedError()

    def score(self, data_handler: DataHandler, metric_fn: Callable) -> tuple[float, Any]:
        """calculates metric on test set and returns metric and useful assets that represent intermediate data"""
        raise NotImplementedError()

    def fit_score(self, data_handler: DataHandler, metric_fn: Callable) -> tuple[float, Any]:
        self.fit(data_handler)
        return self.score(data_handler, metric_fn)

    def clear_cache(self):
        """clear GPU/CPU memory"""
        raise NotImplementedError()
