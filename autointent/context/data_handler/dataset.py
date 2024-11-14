from functools import cached_property
from typing import Any, TypedDict

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import Sequence, concatenate_datasets
from typing_extensions import Self

from autointent.custom_types import LabelType

from .data_loader import DatasetLoader, Intent


class Split:
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    OOS = "oos"


class Sample(TypedDict):
    utterance: str
    label: LabelType | None


class Dataset(HFDatasetDict):
    LABEL_COLUMN = "label"
    UTTERANCE_COLUMN = "utterance"

    def __init__(self, *args: Any, intents: list[Intent], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.intents = intents
        self.create_oos_split()

    @property
    def multilabel(self) -> bool:
        return isinstance(self[Split.TRAIN].features[self.LABEL_COLUMN], Sequence)

    @cached_property
    def n_classes(self) -> int:
        classes = set()
        for label in self[Split.TRAIN][self.LABEL_COLUMN]:
            match label:
                case int():
                    classes.add(label)
                case list():
                    for label_ in label:
                        classes.add(label_)
        return len(classes)

    @classmethod
    def from_dataset_loader(cls, data_loader: DatasetLoader) -> Self:
        splits = data_loader.model_dump(exclude_defaults=True)
        intents = splits.pop("intents", [])
        return cls(
            {
                split_name: HFDataset.from_list(split)
                for split_name, split in splits.items()
            },
            intents=intents,
        )

    def dump(self) -> dict[str, list[dict[str, Any]]]:
        return {split: self[split].to_list() for split in self}


    def create_oos_split(self) -> None:
        oos_splits = self.filter(self._is_oos).values()
        oos_splits = [oos_split for oos_split in oos_splits if oos_split.num_rows]
        if oos_splits:
            splits = self.filter(lambda sample: not self._is_oos(sample))
            for split_name, split in splits.items():
                self[split_name] = split
            self[Split.OOS] = concatenate_datasets(oos_splits)

    def encode_labels(self) -> Self:
        for split_name, split in self.map(self._encode_label).items():
            self[split_name] = split
        return self

    def to_multilabel(self) -> Self:
        for split_name, split in self.map(self._to_multilabel).items():
            self[split_name] = split
        return self

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
