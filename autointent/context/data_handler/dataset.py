from typing import Any

from datasets import Dataset, Sequence, concatenate_datasets
from datasets import DatasetDict as HFDatasetDict
from typing_extensions import Self


class DatasetDict(HFDatasetDict):
    utterance_splits = ("train", "validation", "test")

    @property
    def n_classes(self) -> int:
        classes = set()
        for label in self["train"]["label"]:
            match label:
                case int():
                    classes.add(label)
                case list():
                    for label_ in label:
                        classes.add(label_)
        return len(classes)

    @property
    def multilabel(self) -> bool:
        return isinstance(self["train"].features["label"], Sequence)

    def filter_oos(self) -> Self:
        oos_splits = []
        for split in self.utterance_splits:
            if split in self:
                is_split, oos_split = self._filter_oos(self[split])
                self[split] = is_split
                oos_splits.append(oos_split)
        self["oos"] = concatenate_datasets(oos_splits)
        return self

    def one_hot_labels(self) -> Self:
        for split in self.utterance_splits:
            if split in self:
                self[split] = self[split].map(self._one_hot_label)
        return self

    def to_multilabel(self) -> Self:
        for split in self.utterance_splits:
            if split in self:
                self[split] = self[split].map(self._sample_to_multilabel)
        return self

    def _filter_oos(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        is_indices, oos_indices = [], []
        for idx, label in enumerate(dataset["label"]):
            oos_indices.append(idx) if label is None else is_indices.append(idx)
        return dataset.select(is_indices), dataset.select(oos_indices)

    def _one_hot_label(self, sample: dict[str, Any]) -> dict[str, Any]:
        one_hot_label = [0] * self.n_classes
        if sample["label"] is not None:
            label = [sample["label"]] if isinstance(sample["label"], int) else sample["label"]
            for idx in label:
                one_hot_label[idx] = 1
        sample["label"] = one_hot_label
        return sample

    def _sample_to_multilabel(self, sample: dict[str, Any]) -> dict[str, Any]:
        if isinstance(sample["label"], int):
            sample["label"] = [sample["label"]]
        return sample
