"""Module for defining record schemas for intents and utterances.

This module provides `TypedDict` classes for structured representation of
intent and utterance records, ensuring type safety and clarity.
"""

from typing import TypedDict

from autointent.custom_types import LabelType


class IntentRecord(TypedDict):
    """Represents a record for an intent."""

    intent_id: int
    regexp_full_match: str
    regexp_partial_match: str


class UtteranceRecord(TypedDict):
    """Represents a record for an utterance."""

    utterance: str
    labels: LabelType | tuple[int, ...]
