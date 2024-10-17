from enum import Enum
from functools import cached_property
from typing import Literal

from pydantic import BaseModel


class DatasetType(str, Enum):
    multilabel = "multilabel"
    multiclass = "multiclass"


class Utterance(BaseModel):
    text: str
    label: int | list[int] | None = None

    def one_hot_label(self, n_classes: int) -> list[int]:
        encoding = [0] * n_classes
        label = [self.label] if isinstance(self.label, int) else self.label
        for idx in label:
            encoding[idx] = 1
        return encoding

    @cached_property
    def type(self) -> DatasetType:
        if isinstance(self.label, int):
            return DatasetType.multiclass
        return DatasetType.multilabel

    @cached_property
    def oos(self) -> bool:
        return self.label is None


class Intent(BaseModel):
    id: int
    name: str | None = None
    tags: list[str] = []
    regexp_full_match: list[str] = []
    regexp_partial_match: list[str] = []


class Dataset(BaseModel):
    utterances: list[Utterance]
    intents: list[Intent] = []

    @cached_property
    def type(self) -> DatasetType:
        if all(utterance.type == DatasetType.multilabel for utterance in self.utterances):
            return DatasetType.multilabel
        return DatasetType.multiclass

    @cached_property
    def n_classes(self) -> int:
        classes = set()
        for utterance in self.utterances:
            if utterance.oos:
                continue
            if self.type == DatasetType.multiclass:
                classes.add(utterance.label)
            else:
                for label in utterance.label:
                    classes.add(label)
        return len(classes)
