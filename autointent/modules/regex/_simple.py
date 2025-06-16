"""Module for regular expressions based intent detection."""

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.configs import CrossEncoderConfig, EmbedderConfig
from autointent.context.optimization_info import Artifact
from autointent.custom_types import LabelType, ListOfGenericLabels, ListOfLabels
from autointent.metrics import REGEX_METRICS
from autointent.modules.base import BaseRegex
from autointent.schemas import Intent


class RegexPatternsCompiled(TypedDict):
    """Compiled regex patterns.

    Attributes:
        id: Intent ID
        regex_full_match: Compiled regex patterns for full match
        regex_partial_match: Compiled regex patterns for partial match
    """

    id: int
    regex_full_match: list[re.Pattern[str]]
    regex_partial_match: list[re.Pattern[str]]


class SimpleRegex(BaseRegex):
    """Regular expressions based intent detection module.

    A module that uses regular expressions to detect intents in text utterances.
    Supports both full and partial pattern matching.
    """

    name = "simple"
    supports_multiclass = True
    supports_multilabel = True
    supports_oos = False

    @classmethod
    def from_context(cls, context: Context) -> "SimpleRegex":
        """Initialize from context.

        Args:
            context: Context object containing configuration

        Returns:
            Initialized SimpleRegex instance
        """
        return cls()

    def fit(self, intents: list[Intent]) -> None:
        """Fit the model with intent patterns.

        Args:
            intents: List of intents to fit the model with
        """
        regex_patterns = [
            {
                "id": intent.id,
                "regex_full_match": intent.regex_full_match,
                "regex_partial_match": intent.regex_partial_match,
            }
            for intent in intents
        ]
        self._compile_regex_patterns(regex_patterns)

    def predict(self, utterances: list[str]) -> list[LabelType]:
        """Predict intents for given utterances.

        Args:
            utterances: List of utterances to predict intents for

        Returns:
            List of predicted intent labels
        """
        return [self._predict_single(utterance)[0] for utterance in utterances]

    def predict_with_metadata(
        self,
        utterances: list[str],
    ) -> tuple[list[LabelType], list[dict[str, Any]] | None]:
        """Predict intents for utterances with pattern matching metadata.

        Args:
            utterances: List of utterances to predict intents for

        Returns:
            Tuple containing:
                - List of predicted intent labels
                - List of pattern matching metadata for each utterance
        """
        predictions, metadata = [], []
        for utterance in utterances:
            prediction, matches = self._predict_single(utterance)
            predictions.append(prediction)
            metadata.append(matches)
        return predictions, metadata

    def _match(self, utterance: str, intent_record: RegexPatternsCompiled) -> dict[str, list[str]]:
        """Match utterance with intent record patterns.

        Args:
            utterance: Utterance to match
            intent_record: Intent record containing patterns to match against

        Returns:
            Dictionary containing matched full and partial patterns
        """
        full_matches = [
            pattern.pattern for pattern in intent_record["regex_full_match"] if pattern.fullmatch(utterance) is not None
        ]
        partial_matches = [
            pattern.pattern for pattern in intent_record["regex_partial_match"] if pattern.search(utterance) is not None
        ]
        return {"full_matches": full_matches, "partial_matches": partial_matches}

    def _predict_single(self, utterance: str) -> tuple[LabelType, dict[str, list[str]]]:
        """Predict intent for a single utterance.

        Args:
            utterance: Utterance to predict intent for

        Returns:
            Tuple containing:
                - Predicted intent labels
                - Dictionary of matched patterns
        """
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
        """Score the model using holdout validation.

        Args:
            context: Context containing validation data
            metrics: List of metric names to compute

        Returns:
            Dictionary of computed metric values
        """
        self.fit(context.data_handler.dataset.intents)

        val_utterances = context.data_handler.validation_utterances(0)
        val_labels = context.data_handler.validation_labels(0)

        pred_labels = self.predict(val_utterances)

        chosen_metrics = {name: fn for name, fn in REGEX_METRICS.items() if name in metrics}
        return self.score_metrics_ho((val_labels, pred_labels), chosen_metrics)

    def score_cv(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """Score the model in cross-validation mode.

        Args:
            context: Context containing validation data
            metrics: List of metric names to compute

        Returns:
            Dictionary of computed metric values
        """
        chosen_metrics = {name: fn for name, fn in REGEX_METRICS.items() if name in metrics}

        metrics_calculated, _ = self.score_metrics_cv(
            chosen_metrics, context.data_handler.validation_iterator(), intents=context.data_handler.dataset.intents
        )

        return metrics_calculated

    def score_metrics_cv(
        self,
        metrics_dict: dict[str, Any],
        cv_iterator: Iterable[tuple[list[str], ListOfLabels, list[str], ListOfLabels]],
        intents: list[Intent],
    ) -> tuple[dict[str, float], list[ListOfGenericLabels] | list[npt.NDArray[Any]]]:
        """Score metrics using cross-validation.

        Args:
            metrics_dict: Dictionary of metrics to compute
            cv_iterator: Cross-validation iterator
            intents: intents from the dataset

        Returns:
            Tuple of metrics dictionary and predictions
        """
        metrics_values: dict[str, list[float]] = {name: [] for name in metrics_dict}
        all_val_preds = []

        self.fit(intents)

        for _, _, val_utterances, val_labels in cv_iterator:
            val_preds = self.predict(val_utterances)
            for name, fn in metrics_dict.items():
                metrics_values[name].append(fn(val_labels, val_preds))
            all_val_preds.append(val_preds)

        metrics = {name: float(np.mean(values_list)) for name, values_list in metrics_values.items()}
        return metrics, all_val_preds  # type: ignore[return-value]

    def clear_cache(self) -> None:
        """Clear cached regex patterns."""
        del self.regex_patterns_compiled

    def get_assets(self) -> Artifact:
        """Get model assets.

        Returns:
            Empty Artifact object
        """
        return Artifact()

    def _compile_regex_patterns(self, regex_patterns: list[dict[str, Any]]) -> None:
        """Compile regex patterns with case-insensitive flag."""
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
            for regex_patterns in regex_patterns
        ]

    def dump(self, path: str) -> None:
        serialized = [
            {
                "id": regex_patterns["id"],
                "regex_full_match": [pattern.pattern for pattern in regex_patterns["regex_full_match"]],
                "regex_partial_match": [pattern.pattern for pattern in regex_patterns["regex_partial_match"]],
            }
            for regex_patterns in self.regex_patterns_compiled
        ]

        dump_dir = Path(path)
        dump_dir.mkdir(parents=True, exist_ok=True)
        with (dump_dir / "regex_patterns.json").open("w", encoding="utf-8") as file:
            json.dump(serialized, file, indent=4, ensure_ascii=False)

    @classmethod
    def load(
        cls,
        path: str,
        embedder_config: EmbedderConfig | None = None,
        cross_encoder_config: CrossEncoderConfig | None = None,
    ) -> "SimpleRegex":
        instance = cls()

        with (Path(path) / "regex_patterns.json").open(encoding="utf-8") as file:
            serialized: list[dict[str, Any]] = json.load(file)

        instance._compile_regex_patterns(serialized)  # noqa: SLF001
        return instance
