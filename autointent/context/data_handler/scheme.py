from typing import TypedDict

from autointent.custom_types import LabelType


class IntentRecord(TypedDict):
    intent_id: int
    regexp_full_match: str
    regexp_partial_match: str


class UtteranceRecord(TypedDict):
    utterance: str
    labels: LabelType | tuple[int, ...]
