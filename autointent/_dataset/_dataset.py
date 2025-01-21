"""File with Dataset definition."""

import json
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, TypedDict

from datasets import ClassLabel, Sequence, get_dataset_config_names, load_dataset
from datasets import Dataset as HFDataset

from autointent.custom_types import LabelType, Split
from autointent.schemas import Intent, Tag


class Sample(TypedDict):
    """
    Typed dictionary representing a dataset sample.

    :param str utterance: The text of the utterance.
    :param LabelType | None label: The label associated with the utterance, or None if out-of-scope.
    """

    utterance: str
    label: LabelType | None


class Dataset(dict[str, HFDataset]):
    """
    Represents a dataset with associated metadata and utilities for processing.

    :param args: Positional arguments to initialize the dataset.
    :param intents: List of intents associated with the dataset.
    :param kwargs: Additional keyword arguments to initialize the dataset.
    """

    label_feature = "label"
    utterance_feature = "utterance"

    def __init__(self, *args: Any, intents: list[Intent], **kwargs: Any) -> None:  # noqa: ANN401
        """
        Initialize the dataset.

        :param args: Positional arguments to initialize the dataset.
        :param intents: List of intents associated with the dataset.
        :param kwargs: Additional keyword arguments to initialize the dataset.
        """
        super().__init__(*args, **kwargs)

        self.intents = intents

        self._encoded_labels = False

        if self.multilabel and not self._is_one_hot_encoded():
            self._encode_labels()

    @property
    def multilabel(self) -> bool:
        """
        Check if the dataset is multilabel.

        :return: True if the dataset is multilabel, False otherwise.
        """
        split = Split.TRAIN if Split.TRAIN in self else f"{Split.TRAIN}_0"
        return isinstance(self[split].features[self.label_feature], Sequence)

    def _is_one_hot_encoded(self) -> bool:
        """
        Check the format of labels in multi-label case.

        Dataset labels in multi-label case can be represented either by list of
        integers or by list of zeros and ones
        """
        split = Split.TRAIN if Split.TRAIN in self else f"{Split.TRAIN}_0"
        in_domain_samples = self[split].filter(lambda sample: sample[self.label_feature] is not None)
        return all(self._is_ohe_single(sample[self.label_feature]) for sample in in_domain_samples)

    def _is_ohe_single(self, label: list[int]) -> bool:
        return len(label) == self.n_classes and all(item in [0, 1] for item in label)

    @cached_property
    def n_classes(self) -> int:
        """
        Get the number of classes in the training split.

        :return: Number of classes.
        """
        return len(self.intents)

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> "Dataset":
        """
        Load a dataset from a dictionary mapping.

        :param mapping: Dictionary representing the dataset.
        :return: Initialized Dataset object.
        """
        from ._reader import DictReader

        return DictReader().read(mapping)

    @classmethod
    def from_json(cls, filepath: str | Path) -> "Dataset":
        """
        Load a dataset from a JSON file.

        :param filepath: Path to the JSON file.
        :return: Initialized Dataset object.
        """
        from ._reader import JsonReader

        return JsonReader().read(filepath)

    @classmethod
    def from_hub(cls, repo_id: str) -> "Dataset":
        """
        Load a dataset from a Hugging Face repository.

        :param repo_id: ID of the Hugging Face repository.
        :return: Initialized Dataset object.
        """
        splits, intents = load_dataset(repo_id), []
        if Split.INTENTS in get_dataset_config_names(repo_id):
            intents = load_dataset(repo_id, Split.INTENTS)[Split.INTENTS].to_list()
        return cls(
            splits.items(),
            intents=[Intent.model_validate(intent) for intent in intents],
        )

    def to_multilabel(self) -> "Dataset":
        """
        Convert dataset labels to multilabel format.

        :return: Self, with labels converted to multilabel.
        """
        for split_name, split in self.items():
            self[split_name] = split.map(self._to_multilabel)
        self._encode_labels()
        return self

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """
        Convert the dataset splits and intents to a dictionary of lists.

        :return: A dictionary containing dataset splits and intents as lists of dictionaries.
        """
        mapping = {split_name: split.to_list() for split_name, split in self.items()}
        mapping[Split.INTENTS] = [intent.model_dump() for intent in self.intents]
        return mapping

    def to_json(self, filepath: str | Path) -> None:
        """
        Save the dataset splits and intents to a JSON file.

        :param filepath: The path to the file where the JSON data will be saved.
        """
        path = Path(filepath)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with path.open("w") as file:
            json.dump(self.to_dict(), file, indent=4, ensure_ascii=False)

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push dataset splits to a Hugging Face repository.

        :param repo_id: ID of the Hugging Face repository.
        """
        for split_name, split in self.items():
            split.push_to_hub(repo_id, split=split_name, private=private)

        if self.intents:
            intents = HFDataset.from_list([intent.model_dump() for intent in self.intents])
            intents.push_to_hub(repo_id, config_name=Split.INTENTS, split=Split.INTENTS)

    def get_tags(self) -> list[Tag]:
        """
        Extract unique tags from the dataset's intents.

        :return: List of tags with their associated intent IDs.
        """
        tag_mapping = defaultdict(list)
        for intent in self.intents:
            for tag in intent.tags:
                tag_mapping[tag].append(intent.id)
        return [Tag(name=tag, intent_ids=intent_ids) for tag, intent_ids in tag_mapping.items()]

    def get_n_classes(self, split: str) -> int:
        """
        Calculate the number of unique classes in a given split.

        :param split: The split to analyze.
        :return: Number of unique classes.
        """
        classes = set()
        for label in self[split][self.label_feature]:
            match (label, self._encoded_labels):
                case (int(), _):
                    classes.add(label)
                case (list(), False):
                    for label_ in label:
                        classes.add(label_)
                case (list(), True):
                    for idx, label_ in enumerate(label):
                        if label_:
                            classes.add(idx)
        return len(classes)

    def _encode_labels(self) -> "Dataset":
        """
        Encode dataset labels into one-hot or multilabel format.

        :return: Self, with labels encoded.
        """
        for split_name, split in self.items():
            self[split_name] = split.map(self._encode_label)
        self._encoded_labels = True
        return self

    def _to_multilabel(self, sample: Sample) -> Sample:
        """
        Convert a sample's label to multilabel format.

        :param sample: The sample to process.
        :return: Sample with label in multilabel format.
        """
        if isinstance(sample["label"], int):
            sample["label"] = [sample["label"]]
        return sample

    def _encode_label(self, sample: Sample) -> Sample:
        """
        Encode a sample's label as a one-hot vector.

        :param sample: The sample to encode.
        :return: Sample with encoded label.
        """
        one_hot_label = [0] * self.n_classes
        match sample["label"]:
            case int():
                one_hot_label[sample["label"]] = 1
            case list():
                for idx in sample["label"]:
                    one_hot_label[idx] = 1
        sample["label"] = one_hot_label
        return sample

    def _cast_label_feature(self) -> None:
        """Cast the label feature of the dataset to the appropriate type."""
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
