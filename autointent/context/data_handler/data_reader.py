import json
from pathlib import Path

from pydantic import BaseModel
from typing_extensions import Self

from autointent.custom_types import LabelType


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


class Dataset(BaseModel):
    train: list[Sample]
    validation: list[Sample] = []
    test: list[Sample] = []
    intents: list[Intent] = []

    @classmethod
    def from_json(cls, filepath: str | Path) -> Self:
        with Path.open(filepath) as file:
            return cls.model_validate(json.load(file))


class JsonReader:
    def read(self, filepath: str | Path) -> Dataset:
        with Path.open(filepath) as file:
            return Dataset.model_validate(json.load(file))
