import re
from enum import Enum
from functools import cached_property
from typing import Any

from pydantic import BaseModel, field_validator, model_serializer

from autointent.custom_types import LABEL_TYPE


class UtteranceType(str, Enum):
    oos = "oos"
    multilabel = "multilabel"
    multiclass = "multiclass"


class DatasetType(str, Enum):
    multilabel = "multilabel"
    multiclass = "multiclass"


class Utterance(BaseModel):
    text: str
    label: LABEL_TYPE | None = None

    def one_hot_label(self, n_classes: int) -> LABEL_TYPE:
        encoding = [0] * n_classes
        label = [self.label] if isinstance(self.label, int) else self.label
        if label is None:
            msg = "Cannot one-hot encode OOS utterance"
            raise ValueError(msg)
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


class RegExpPatterns(BaseModel):
    id: int
    regexp_full_match: list[re.Pattern[str]]
    regexp_partial_match: list[re.Pattern[str]]

    @field_validator("regexp_full_match", "regexp_partial_match", mode="before")
    @classmethod
    def _compile(cls, patterns: list[str]) -> list[re.Pattern[str]]:
        return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]

    @model_serializer
    def _serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "regexp_full_match": self._serialize_patterns(self.regexp_full_match),
            "regexp_partial_match": self._serialize_patterns(self.regexp_partial_match),
        }

    def _serialize_patterns(self, patterns: list[re.Pattern[str]]) -> list[str]:
        return [pattern.pattern for pattern in patterns]


class Intent(BaseModel):
    id: int
    name: str | None = None
    tags: list[str] = []
    regexp_full_match: list[str] = []
    regexp_partial_match: list[str] = []

    def get_regexp_patterns(self) -> RegExpPatterns:
        return RegExpPatterns(
            id=self.id,
            regexp_full_match=self.regexp_full_match,
            regexp_partial_match=self.regexp_partial_match,
        )


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
            elif isinstance(utterance.label, list):
                for label in utterance.label:
                    classes.add(label)
        return len(classes)

    def to_multilabel(self) -> "Dataset":
        return Dataset(utterances=[utterance.to_multilabel() for utterance in self.utterances], intents=self.intents)
