from enum import Enum
from functools import cached_property

from pydantic import BaseModel


class UtteranceType(str, Enum):
    oos = "oos"
    multilabel = "multilabel"
    multiclass = "multiclass"


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
    def type(self) -> UtteranceType:
        if self.label is None:
            return UtteranceType.oos
        if isinstance(self.label, int):
            return UtteranceType.multiclass
        return UtteranceType.multilabel

    @cached_property
    def oos(self) -> bool:
        return self.label is None

    def to_multilabel(self) -> "Utterance":
        if self.type in {UtteranceType.multilabel, UtteranceType.oos}:
            return self
        return Utterance(text=self.text, label=[self.label])


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
        if all(utterance.type in {UtteranceType.multiclass, UtteranceType.oos} for utterance in self.utterances):
            return DatasetType.multiclass
        return DatasetType.multilabel  # TODO add proper dataset type validation

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

    def to_multilabel(self) -> "Dataset":
        return Dataset(
            utterances=[utterance.to_multilabel() for utterance in self.utterances],
            intents=self.intents
        )
