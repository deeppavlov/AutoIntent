from pydantic import BaseModel

from autointent.custom_types import LabelType


class Tag(BaseModel):
    name: str
    intent_ids: list[int]


class Sample(BaseModel):
    utterance: str
    label: LabelType | None = None


class Intent(BaseModel):
    id: int
    name: str | None = None
    tags: list[str] = []
    regexp_full_match: list[str] = []
    regexp_partial_match: list[str] = []
    description: str | None = None
