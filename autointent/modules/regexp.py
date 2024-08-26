import re
from copy import deepcopy
from typing import Any, Callable

from .base import DataHandler, Module


class RegExp(Module):
    def fit(self, data_handler: DataHandler):
        regexp_patterns = deepcopy(data_handler.regexp_patterns)
        for dct in regexp_patterns:
            dct["regexp_full_match"] = [
                re.compile(ptn, flags=re.IGNORECASE)
                for ptn in dct["regexp_full_match"]
            ]
            dct["regexp_partial_match"] = [
                re.compile(ptn, flags=re.IGNORECASE)
                for ptn in dct["regexp_partial_match"]
            ]
        self.regexp_patterns = regexp_patterns

    def predict(self, utterances: list[str]) -> list[set]:
        return [self._predict_single(ut) for ut in utterances]

    def _match(self, text: str, intent_record: dict):
        full_match = any(ptn.fullmatch(text) for ptn in intent_record["regexp_full_match"])
        if full_match:
            return True
        partial_match = any(ptn.match(text) for ptn in intent_record["regexp_partial_match"])
        return partial_match

    def _predict_single(self, utterance: str):
        return set(
            intent_record["intent_id"]
            for intent_record in self.regexp_patterns
            if self._match(utterance, intent_record)
        )

    def score(self, data_handler: DataHandler, metric_fn: Callable) -> tuple[float, Any]:
        # TODO add parameter to a whole pipeline (or just to regexp module): whether or not to omit utterances on next stages if they were detected with regexp module
        assets = dict(
            test_matches=self.predict(data_handler.utterances_test),
            oos_matches=None
            if len(data_handler.oos_utterances) == 0
            else self.predict(data_handler.oos_utterances),
        )

        metric_value = metric_fn(data_handler.labels_test, assets["test_matches"])

        return metric_value, assets

    def clear_cache(self):
        del self.regexp_patterns
