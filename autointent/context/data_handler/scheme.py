from typing import TypedDict


class IntentRecord(TypedDict):
    intent_id: str
    regexp_full_match: str
    regexp_partial_match: str


class UtteranceRecord(TypedDict):
    utterance: str
    labels: list[int] | tuple[int, ...]
