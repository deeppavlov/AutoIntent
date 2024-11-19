from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, TypedDict

from datasets import ClassLabel, Sequence, concatenate_datasets
from datasets import Dataset as HFDataset
from typing_extensions import Self

from autointent.custom_types import LabelType

from .schemas import Tag
from .validation import Intent


class Split:
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    OOS = "oos"


class Sample(TypedDict):
    utterance: str
    label: LabelType | None


class Dataset(dict[str, HFDataset]):
    label_feature = "label"
    utterance_feature = "utterance"

    def __init__(self, *args: Any, intents: list[Intent], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.intents = intents

        oos_split = self._create_oos_split()
        if oos_split is not None:
            self[Split.OOS] = oos_split

        self._cast_label_feature()

    @property
    def multilabel(self) -> bool:
        return isinstance(self[Split.TRAIN].features[self.label_feature], Sequence)

    @cached_property
    def n_classes(self) -> int:
        classes = set()
        for label in self[Split.TRAIN][self.label_feature]:
            match label:
                case int():
                    classes.add(label)
                case list():
                    for label_ in label:
                        classes.add(label_)
        return len(classes)

    @classmethod
    def from_json(cls, filepath: str | Path) -> Self:
        from .reader import JsonReader
        return JsonReader().read(filepath)

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> Self:
        from .reader import DictReader
        return DictReader().read(mapping)

    def dump(self) -> dict[str, list[dict[str, Any]]]:
        return {split_name: split.to_list() for split_name, split in self.items()}

    def encode_labels(self) -> Self:
        for split_name, split in self.items():
            self[split_name] = split.map(self._encode_label)
        return self

    def to_multilabel(self) -> Self:
        for split_name, split in self.items():
            self[split_name] = split.map(self._to_multilabel)
        self._cast_label_column()
        return self

    def get_tags(self) -> list[Tag]:
        tag_mapping = defaultdict(list)
        for intent in self.intents:
            for tag in intent.tags:
                tag_mapping[tag].append(intent.id)

        return [
            Tag(name=tag, intent_ids=intent_ids)
            for tag, intent_ids in tag_mapping.items()
        ]

    def _is_oos(self, sample: Sample) -> bool:
        return sample["label"] is None

    def _to_multilabel(self, sample: Sample) -> Sample:
        if isinstance(sample["label"], int):
            sample["label"] = [sample["label"]]
        return sample

    def _encode_label(self, sample: Sample) -> Sample:
        one_hot_label = [0] * self.n_classes
        match sample["label"]:
            case int():
                one_hot_label[sample["label"]] = 1
            case list():
                for idx in sample["label"]:
                    one_hot_label[idx] = 1
        sample["label"] = one_hot_label
        return sample

    def _create_oos_split(self) -> HFDataset | None:
        oos_splits = [split.filter(self._is_oos) for split in self.values()]
        oos_splits = [oos_split for oos_split in oos_splits if oos_split.num_rows]
        if oos_splits:
            splits = self.filter(lambda sample: not self._is_oos(sample))
            for split_name, split in splits.items():
                self[split_name] = split
            return concatenate_datasets(oos_splits)
        return None

    def _cast_label_feature(self) -> None:
        for split_name, split in self.items():
            new_features = split.features.copy()
            if self.multilabel:
                new_features[self.label_feature] = Sequence(
                    ClassLabel(num_classes=self.n_classes),
                )
            else:
                new_features[self.label_feature] = ClassLabel(
                    num_classes=self.n_classes,
                )
            self[split_name] = split.cast(new_features)
