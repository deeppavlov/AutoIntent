import re
from typing import Any, TypedDict

from typing_extensions import Self

from autointent import Context
from autointent.context.optimization_info.data_models import Artifact
from autointent.custom_types import LABEL_TYPE
from autointent.metrics.regexp import RegexpMetricFn

from .base import Module
from autointent.context.data_handler.data_handler import RegexPatterns


class RegexPatternsCompiled(TypedDict):
    id: int
    regexp_full_match: list[re.Pattern[str]]
    regexp_partial_match: list[re.Pattern[str]]



class RegExp(Module):
    def __init__(self, regexp_patterns: list[RegexPatterns]) -> None:
        self.regexp_patterns = regexp_patterns

    @classmethod
    def from_context(cls, context: Context, **kwargs: dict[str, Any]) -> Self:
        return cls(
            regexp_patterns=context.data_handler.regexp_patterns,
        )

    def fit(self, utterances: list[str], labels: list[LABEL_TYPE], **kwargs: dict[str, Any]) -> None:
        self.regexp_patterns_compiled: list[RegexPatternsCompiled] = [
            {
                "id": dct["id"],
                "regexp_full_match": [re.compile(ptn, flags=re.IGNORECASE) for ptn in dct["regexp_full_match"]],
                "regexp_partial_match": [re.compile(ptn, flags=re.IGNORECASE) for ptn in dct["regexp_partial_match"]],
            }
            for dct in self.regexp_patterns
        ]

    def predict(self, utterances: list[str]) -> list[LABEL_TYPE]:
        return [list(self._predict_single(ut)) for ut in utterances]

    def _match(self, text: str, intent_record: RegexPatternsCompiled) -> bool:
        full_match = any(ptn.fullmatch(text) for ptn in intent_record["regexp_full_match"])
        if full_match:
            return True
        return any(ptn.match(text) for ptn in intent_record["regexp_partial_match"])

    def _predict_single(self, utterance: str) -> set[int]:
        # todo test this
        return {intent_record["id"] for intent_record in self.regexp_patterns_compiled if self._match(utterance, intent_record)}

    def score(self, context: Context, metric_fn: RegexpMetricFn) -> float:
        # TODO add parameter to a whole pipeline (or just to regexp module):
        # whether or not to omit utterances on next stages if they were detected with regexp module
        assets = {
            "test_matches": list(self.predict(context.data_handler.utterances_test)),
            "oos_matches": None
            if len(context.data_handler.oos_utterances) == 0
            else self.predict(context.data_handler.oos_utterances),
        }
        if assets["test_matches"] is None:
            msg = "no matches found"
            raise ValueError(msg)
        return metric_fn(context.data_handler.labels_test, assets["test_matches"])

    def clear_cache(self) -> None:
        del self.regexp_patterns

    def get_assets(self) -> Artifact:
        return Artifact()

    def load(self, path: str) -> None:
        pass

    def dump(self, path: str) -> None:
        pass
