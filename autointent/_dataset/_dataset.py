"""File with Dataset definition."""

import json
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, TypedDict

from datasets import Dataset as HFDataset
from datasets import Sequence, get_dataset_config_names, load_dataset

from autointent.custom_types import LabelWithOOS, Split
from autointent.schemas import Intent, Tag


class Sample(TypedDict):
    """
    Typed dictionary representing a dataset sample.

    :param utterance: The text of the utterance.
    :param label: The label associated with the utterance, or None if out-of-scope.
    """

    utterance: str
    label: LabelWithOOS


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

    @property
    def multilabel(self) -> bool:
        """
        Check if the dataset is multilabel.

        :return: True if the dataset is multilabel, False otherwise.
        """
        split = Split.TRAIN if Split.TRAIN in self else f"{Split.TRAIN}_0"
        return isinstance(self[split].features[self.label_feature], Sequence)

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
        from ._reader import DictReader

        splits = load_dataset(repo_id)
        mapping = dict(**splits)
        if Split.INTENTS in get_dataset_config_names(repo_id):
            mapping["intents"] = load_dataset(repo_id, Split.INTENTS)[Split.INTENTS].to_list()

        return DictReader().read(mapping)

    def to_multilabel(self) -> "Dataset":
        """
        Convert dataset labels to multilabel format.

        :return: Self, with labels converted to multilabel.
        """
        for split_name, split in self.items():
            self[split_name] = split.map(self._to_multilabel)
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
        :param private: Whether the repository is private
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
            match label:
                case int():
                    classes.add(label)
                case list():
                    for idx, label_ in enumerate(label):
                        if label_:
                            classes.add(idx)
        return len(classes)

    def _to_multilabel(self, sample: Sample) -> Sample:
        """
        Convert a sample's label to multilabel format.

        :param sample: The sample to process.
        :return: Sample with label in multilabel format.
        """
        if isinstance(sample["label"], int):
            ohe_vector = [0] * self.n_classes
            ohe_vector[sample["label"]] = 1
            sample["label"] = ohe_vector
        return sample
