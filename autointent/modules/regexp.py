import re
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from autointent import Context
from autointent.context.optimization_info.data_models import Artifact

from .base import Module


class RegExp(Module):
    def fit(self, context: Context) -> None:
        regexp_patterns = deepcopy(context.data_handler.regexp_patterns)
        for dct in regexp_patterns:
            dct["regexp_full_match"] = [re.compile(ptn, flags=re.IGNORECASE) for ptn in dct["regexp_full_match"]]
            dct["regexp_partial_match"] = [re.compile(ptn, flags=re.IGNORECASE) for ptn in dct["regexp_partial_match"]]
        self.regexp_patterns = regexp_patterns

    def predict(self, utterances: list[str]) -> list[list[int]]:
        return [list(self._predict_single(ut)) for ut in utterances]

    def _match(self, text: str, intent_record: dict) -> bool:
        full_match = any(ptn.fullmatch(text) for ptn in intent_record["regexp_full_match"])
        if full_match:
            return True
        return any(ptn.match(text) for ptn in intent_record["regexp_partial_match"])

    def _predict_single(self, utterance: str) -> set[int]:
        return {
            intent_record["intent_id"]
            for intent_record in self.regexp_patterns
            if self._match(utterance, intent_record)
        }

    def score(self, context: Context, metric_fn: Callable) -> tuple[float, Any]:
        # TODO add parameter to a whole pipeline (or just to regexp module):
        # whether or not to omit utterances on next stages if they were detected with regexp module
        assets = {
            "test_matches": list(self.predict(context.data_handler.utterances_test)),
            "oos_matches": None
            if len(context.data_handler.oos_utterances) == 0
            else self.predict(context.data_handler.oos_utterances),
        }

        metric_value = metric_fn(context.data_handler.labels_test, assets["test_matches"])

        return metric_value, assets

    def clear_cache(self) -> None:
        del self.regexp_patterns

    def get_assets(self) -> Artifact:
        return None
