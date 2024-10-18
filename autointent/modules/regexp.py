import re
from typing import Any
from typing_extensions import override

from autointent import Context
from autointent.context.optimization_info.data_models import Artifact
from autointent.metrics.regexp import RegexpMetricFn

from .base import Module


class RegExp(Module):
    def fit(self, context: Context) -> None:
        self.regexp_patterns = [
            {
                "regexp_full_match": [re.compile(ptn, flags=re.IGNORECASE) for ptn in dct["regexp_full_match"]],
                "regexp_partial_match": [re.compile(ptn, flags=re.IGNORECASE) for ptn in dct["regexp_partial_match"]],
            }
            for dct in context.data_handler.regexp_patterns
        ]

    def predict(self, utterances: list[str]) -> list[list[int]]:
        return [list(self._predict_single(ut)) for ut in utterances]

    def _match(self, text: str, intent_record: dict[str, Any]) -> bool:
        full_match = any(ptn.fullmatch(text) for ptn in intent_record["regexp_full_match"])
        if full_match:
            return True
        return any(ptn.match(text) for ptn in intent_record["regexp_partial_match"])

    def _predict_single(self, utterance: str) -> set[int]:
        return {
            intent_record["intent_id"]  # type: ignore[misc]
            for intent_record in self.regexp_patterns
            if self._match(utterance, intent_record)
        }

    def score(self, context: Context, metric_fn: RegexpMetricFn) -> float:  # type: ignore[override]
        # TODO add parameter to a whole pipeline (or just to regexp module):
        # whether or not to omit utterances on next stages if they were detected with regexp module
        assets = {
            "test_matches": list(self.predict(context.data_handler.utterances_test)),
            "oos_matches": None
            if len(context.data_handler.oos_utterances) == 0
            else self.predict(context.data_handler.oos_utterances),
        }
        if assets["test_matches"] is None:
            raise ValueError("no matches found")
        return metric_fn(context.data_handler.labels_test, assets["test_matches"])

    def clear_cache(self) -> None:
        del self.regexp_patterns

    def get_assets(self) -> Artifact:
        return Artifact()
