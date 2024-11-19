import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, model_validator
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

    @model_validator(mode="after")
    def sort_intents_by_id(self) -> Self:
        self.intents = sorted(self.intents, key=lambda intent: intent.id)
        return self


class Reader(Protocol):
    @abstractmethod
    def read(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


class DictReader:
    def read(self, mapping: dict[str, Any]) -> dict[str, Any]:
        return Dataset.model_validate(mapping).model_dump(exclude_defaults=True)


class JsonReader:
    def read(self, filepath: str | Path) -> dict[str, Any]:
        with Path.open(filepath) as file:
            return Dataset.model_validate(json.load(file)).model_dump(exclude_defaults=True)
