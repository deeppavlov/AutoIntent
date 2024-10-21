from typing import TypedDict

from autointent.custom_types import LABEL_TYPE


class IntentRecord(TypedDict):
    intent_id: int
    regexp_full_match: str
    regexp_partial_match: str


class UtteranceRecord(TypedDict):
    utterance: str
    labels: LABEL_TYPE | tuple[int, ...]
