from typing import Callable

from ..data_handler import DataHandler


class Module:
    def fit(self, data_handler: DataHandler):
        pass

    def score(self, data_handler: DataHandler, metric_fn: Callable) -> float:
        pass

    def fit_score(self, data_handler: DataHandler, metric_fn: Callable) -> float:
        self.fit(data_handler)
        return self.score(data_handler, metric_fn)
