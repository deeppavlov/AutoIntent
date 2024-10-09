from enum import Enum
from functools import cached_property

from pydantic import BaseModel


class DatasetType(str, Enum):
    multilabel = "multilabel"
    multiclass = "multiclass"


class Utterance(BaseModel):
    text: str
    label: list[int] | int

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
        if self.type == DatasetType.multiclass:
            return self.label == -1
        return not self.label


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
        if all(utterance.type == DatasetType.multiclass for utterance in self.utterances):
            return DatasetType.multiclass
        return DatasetType.multilabel
