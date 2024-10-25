import json
from pathlib import Path

from autointent import Context
from autointent.context.data_handler.schemas import RegExpPatterns
from autointent.context.optimization_info.data_models import Artifact
from autointent.custom_types import LABEL_TYPE
from autointent.metrics.regexp import RegexpMetricFn

from .base import Module


class RegExp(Module):
    metadata_dict_name: str = "metadata.json"


    def fit(self, context: Context) -> None:
        self.regexp_patterns = context.data_handler.regexp_patterns

    def predict(self, utterances: list[str]) -> list[LABEL_TYPE]:
        return [list(self._predict(utterance)) for utterance in utterances]

    def _match(self, utterance: str, patterns: RegExpPatterns) -> bool:
        full_match = any(pattern.fullmatch(utterance) for pattern in patterns.regexp_full_match)
        partial_match = any(pattern.match(utterance) for pattern in patterns.regexp_partial_match)
        return full_match or partial_match

    def _predict(self, utterance: str) -> set[int]:
        # TODO testing
        return {
            regexp_patterns.id
            for regexp_patterns in self.regexp_patterns
            if self._match(utterance, regexp_patterns)
        }

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

    def dump(self, path: str) -> None:
        dump_dir = Path(path)

        metadata = [pattern.model_dump() for pattern in self.regexp_patterns]

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(metadata, file, indent=4)

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            metadata =  json.load(file)

        self.regexp_patterns = [RegExpPatterns.model_validate(patterns) for patterns in metadata]
