"""Module for regular expressions based intent detection."""

import re
from typing import Any, TypedDict

from autointent import Context
from autointent.context.data_handler._data_handler import RegexPatterns
from autointent.context.optimization_info import Artifact
from autointent.custom_types import LabelType
from autointent.metrics import REGEX_METRICS
from autointent.modules.abc import BaseRegex
from autointent.schemas import Intent


class RegexPatternsCompiled(TypedDict):
    """Compiled regex patterns."""

    id: int
    """Intent ID."""
    regex_full_match: list[re.Pattern[str]]
    """Compiled regex patterns for full match."""
    regex_partial_match: list[re.Pattern[str]]
    """Compiled regex patterns for partial match."""


class Regex(BaseRegex):
    """Regular expressions based intent detection module."""

    name = "regex"

    @classmethod
    def from_context(cls, context: Context) -> "Regex":
        """Initialize from context."""
        return cls()

    def fit(self, intents: list[Intent]) -> None:
        """
        Fit the model.

        :param intents: Intents to fit
        """
        self.regex_patterns = [
            RegexPatterns(
                id=intent.id,
                regex_full_match=intent.regex_full_match,
                regex_partial_match=intent.regex_partial_match,
            )
            for intent in intents
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
            pattern.pattern for pattern in intent_record["regex_full_match"] if pattern.fullmatch(utterance) is not None
        ]
        partial_matches = [
            pattern.pattern for pattern in intent_record["regex_partial_match"] if pattern.search(utterance) is not None
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
        for intent_record in self.regex_patterns_compiled:
            intent_matches = self._match(utterance, intent_record)
            if intent_matches["full_matches"] or intent_matches["partial_matches"]:
                prediction.add(intent_record["id"])
            matches["full_matches"].extend(intent_matches["full_matches"])
            matches["partial_matches"].extend(intent_matches["partial_matches"])
        return list(prediction), matches

    def score_ho(self, context: Context, metrics: list[str]) -> dict[str, float]:
        self.fit(context.data_handler.dataset.intents)

        val_utterances = context.data_handler.validation_utterances(0)
        val_labels = context.data_handler.validation_labels(0)

        pred_labels = self.predict(val_utterances)

        chosen_metrics = {name: fn for name, fn in REGEX_METRICS.items() if name in metrics}
        return self.score_metrics_ho((val_labels, pred_labels), chosen_metrics)

    def score_cv(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """
        Evaluate the scorer on a test set and compute the specified metric.

        :param context: Context containing test set and other data.
        :param split: Target split
        :return: Computed metrics value for the test set or error code of metrics
        """
        chosen_metrics = {name: fn for name, fn in REGEX_METRICS.items() if name in metrics}

        metrics_calculated, _ = self.score_metrics_cv(chosen_metrics, context.data_handler.validation_iterator())

        return metrics_calculated

    def clear_cache(self) -> None:
        """Clear cache."""
        del self.regex_patterns

    def get_assets(self) -> Artifact:
        """Get assets."""
        return Artifact()

    def _compile_regex_patterns(self) -> None:
        """Compile regex patterns."""
        self.regex_patterns_compiled = [
            RegexPatternsCompiled(
                id=regex_patterns["id"],
                regex_full_match=[
                    re.compile(pattern, flags=re.IGNORECASE) for pattern in regex_patterns["regex_full_match"]
                ],
                regex_partial_match=[
                    re.compile(ptn, flags=re.IGNORECASE) for ptn in regex_patterns["regex_partial_match"]
                ],
            )
            for regex_patterns in self.regex_patterns
        ]
