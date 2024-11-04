from enum import Enum
from functools import cached_property
from typing import Any

import datasets
from pydantic import BaseModel
from typing_extensions import Self

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


class Intent(BaseModel):
    id: int
    name: str | None = None
    tags: list[str] = []
    regexp_full_match: list[str] = []
    regexp_partial_match: list[str] = []
    description: str | None = None


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

    @classmethod
    def from_datasets(
        cls,
        dataset_name: str,
        split: str = "train",
        utterances_kwargs: dict[str, Any] | None = None,
        intents_kwargs: dict[str, Any] | None = None,
        # tags_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        configs = datasets.get_dataset_config_names(dataset_name)

        utterances = []
        intents = []
        if "utterances" in configs:
            utterance_ds = datasets.load_dataset(dataset_name, name="utterances", split=split,
                                                 **(utterances_kwargs or {}))
            utterances = [Utterance(**item) for item in utterance_ds]
        # tags = []
        # if "tags" in configs:
        #     tags_ds = datasets.load_dataset(dataset_name, name="tags", split=split, **(tags_kwargs or {}))
        if "intents" in configs:
            intents_ds = datasets.load_dataset(dataset_name, name="intents", split=split, **(intents_kwargs or {}))
            intents = [Intent(**item) for item in intents_ds]
        return Dataset(utterances=utterances, intents=intents)

    def push_to_hub(self, dataset_name: str, split: str = "train") -> None:
        utterances_ds = datasets.Dataset.from_list([utterance.model_dump() for utterance in self.utterances])
        intents_ds = datasets.Dataset.from_list([intent.model_dump() for intent in self.intents])
        utterances_ds.push_to_hub(dataset_name, config_name="utterances", split=split)
        intents_ds.push_to_hub(dataset_name, config_name="intents", split=split)
