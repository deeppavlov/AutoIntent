"""Module for regular expressions based intent detection."""

import json
import re
from pathlib import Path
from typing import Any, Literal, TypedDict

from typing_extensions import Self

from autointent import Context
from autointent.context.data_handler._data_handler import RegexPatterns
from autointent.context.data_handler._schemas import Intent
from autointent.context.optimization_info import Artifact
from autointent.custom_types import LabelType
from autointent.metrics.regexp import RegexpMetricFn
from autointent.modules.abc import Module


class RegexPatternsCompiled(TypedDict):
    """Compiled regex patterns."""

    id: int
    """Intent ID."""
    regexp_full_match: list[re.Pattern[str]]
    """Compiled regex patterns for full match."""
    regexp_partial_match: list[re.Pattern[str]]
    """Compiled regex patterns for partial match."""


class RegExp(Module):
    """Regular expressions based intent detection module."""

    @classmethod
    def from_context(cls, context: Context) -> Self:
        """Initialize from context."""
        return cls()

    def fit(self, intents: list[dict[str, Any]]) -> None:
        """
        Fit the model.

        :param intents: Intents to fit
        """
        intents_parsed = [Intent(**dct) for dct in intents]
        self.regexp_patterns = [
            RegexPatterns(
                id=intent.id,
                regexp_full_match=intent.regexp_full_match,
                regexp_partial_match=intent.regexp_partial_match,
            )
            for intent in intents_parsed
        ]
        self._compile_regex_patterns()

    def predict(self, utterances: list[str]) -> list[LabelType]:
        """
        Predict intents for utterances.

        :param utterances: Utterances to predict
        """
        return [self._predict_single(utterance)[0] for utterance in utterances]

    def predict_with_metadata(
        self,
        utterances: list[str],
    ) -> tuple[list[LabelType], list[dict[str, Any]] | None]:
        """
        Predict intents for utterances with metadata.

        :param utterances: Utterances to predict
        """
        predictions, metadata = [], []
        for utterance in utterances:
            prediction, matches = self._predict_single(utterance)
            predictions.append(prediction)
            metadata.append(matches)
        return predictions, metadata

    def _match(self, utterance: str, intent_record: RegexPatternsCompiled) -> dict[str, list[str]]:
        """
        Match utterance with intent record.

        :param utterance: Utterance to match
        :param intent_record: Intent record to match
        """
        full_matches = [
            pattern.pattern
            for pattern in intent_record["regexp_full_match"]
            if pattern.fullmatch(utterance) is not None
        ]
        partial_matches = [
            pattern.pattern
            for pattern in intent_record["regexp_partial_match"]
            if pattern.search(utterance) is not None
        ]
        return {"full_matches": full_matches, "partial_matches": partial_matches}

    def _predict_single(self, utterance: str) -> tuple[LabelType, dict[str, list[str]]]:
        """
        Predict intent for a single utterance.

        :param utterance: Utterance to predict
        """
        # todo test this
        prediction = set()
        matches: dict[str, list[str]] = {"full_matches": [], "partial_matches": []}
        for intent_record in self.regexp_patterns_compiled:
            intent_matches = self._match(utterance, intent_record)
            if intent_matches["full_matches"] or intent_matches["partial_matches"]:
                prediction.add(intent_record["id"])
            matches["full_matches"].extend(intent_matches["full_matches"])
            matches["partial_matches"].extend(intent_matches["partial_matches"])
        return list(prediction), matches

    def score(
        self,
        context: Context,
        split: Literal["validation", "test"],
        metric_fn: RegexpMetricFn,
    ) -> float:
        """
        Calculate metric on test set and return metric value.

        :param context: Context to score
        :param metric_fn: Metric function
        """
        # TODO add parameter to a whole pipeline (or just to regexp module):
        # whether or not to omit utterances on next stages if they were detected with regexp module
        assets = {
            "test_matches": list(self.predict(context.data_handler.test_utterances())),
            "oos_matches": None
            if not context.data_handler.has_oos_samples()
            else self.predict(context.data_handler.oos_utterances(2)),
        }
        if assets["test_matches"] is None:
            msg = "no matches found"
            raise ValueError(msg)
        return metric_fn(context.data_handler.test_labels(), assets["test_matches"])

    def clear_cache(self) -> None:
        """Clear cache."""
        del self.regexp_patterns

    def get_assets(self) -> Artifact:
        """Get assets."""
        return Artifact()

    def load(self, path: str) -> None:
        """
        Load data from dump.

        :param path: Path to load
        """
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            self.regexp_patterns = json.load(file)

        self._compile_regex_patterns()

    def dump(self, path: str) -> None:
        """
        Dump all data needed for inference.

        :param path: Path to dump
        """
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.regexp_patterns, file, indent=4)

    def _compile_regex_patterns(self) -> None:
        """Compile regex patterns."""
        self.regexp_patterns_compiled = [
            RegexPatternsCompiled(
                id=regexp_patterns["id"],
                regexp_full_match=[
                    re.compile(pattern, flags=re.IGNORECASE) for pattern in regexp_patterns["regexp_full_match"]
                ],
                regexp_partial_match=[
                    re.compile(ptn, flags=re.IGNORECASE) for ptn in regexp_patterns["regexp_partial_match"]
                ],
            )
            for regexp_patterns in self.regexp_patterns
        ]
