import re
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from .base import Context, Module


class RegExp(Module):
    def fit(self, context: Context):
        regexp_patterns = deepcopy(context.data_handler.regexp_patterns)
        for dct in regexp_patterns:
            dct["regexp_full_match"] = [re.compile(ptn, flags=re.IGNORECASE) for ptn in dct["regexp_full_match"]]
            dct["regexp_partial_match"] = [re.compile(ptn, flags=re.IGNORECASE) for ptn in dct["regexp_partial_match"]]
        self.regexp_patterns = regexp_patterns

    def predict(self, utterances: list[str]) -> list[set]:
        return [self._predict_single(ut) for ut in utterances]

    def _match(self, text: str, intent_record: dict):
        full_match = any(ptn.fullmatch(text) for ptn in intent_record["regexp_full_match"])
        if full_match:
            return True
        return any(ptn.match(text) for ptn in intent_record["regexp_partial_match"])

    def _predict_single(self, utterance: str):
        return {
            intent_record["intent_id"]
            for intent_record in self.regexp_patterns
            if self._match(utterance, intent_record)
        }

    def score(self, context: Context, metric_fn: Callable) -> tuple[float, Any]:
        # TODO add parameter to a whole pipeline (or just to regexp module): whether or not to omit utterances on next stages if they were detected with regexp module
        assets = {
            "test_matches": self.predict(context.data_handler.utterances_test),
            "oos_matches": None
            if len(context.data_handler.oos_utterances) == 0
            else self.predict(context.data_handler.oos_utterances),
        }

        metric_value = metric_fn(context.data_handler.labels_test, assets["test_matches"])

        return metric_value, assets

    def clear_cache(self):
        del self.regexp_patterns
