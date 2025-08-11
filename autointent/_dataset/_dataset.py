"""Defines the Dataset class and related utilities for handling datasets."""

import json
import logging
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, TypedDict

from datasets import Dataset as HFDataset
from datasets import Sequence, get_dataset_config_names, load_dataset

from autointent.custom_types import LabelWithOOS, Split
from autointent.schemas import Intent, Tag

logger = logging.getLogger(__name__)


class Sample(TypedDict):
    """Represents a sample in the dataset.

    Attributes:
        utterance: The text of the utterance.
        label: The label associated with the utterance, or None if it is out-of-scope.
    """

    utterance: str
    label: LabelWithOOS


class Dataset(dict[str, HFDataset]):
    """Represents a dataset with associated metadata and utilities for processing.

    This class extends a dictionary where the keys represent dataset splits (e.g., 'train', 'test'),
    and the values are Hugging Face datasets.
    """

    label_feature: str = "label"
    """The feature name corresponding to labels in the dataset."""

    utterance_feature: str = "utterance"
    """The feature name corresponding to utterances in the dataset"""

    has_descriptions: bool
    """Whether the dataset includes descriptions for intents."""

    intents: list[Intent]
    """All metadata about intents used in this dataset."""

    def __init__(self, *args: Any, intents: list[Intent], **kwargs: Any) -> None:  # noqa: ANN401
        """Initializes the dataset.

        Args:
            *args: Positional arguments used for dataset initialization.
            intents: A list of intents associated with the dataset.
            **kwargs: Additional keyword arguments used for dataset initialization.
        """
        super().__init__(*args, **kwargs)

        self.intents = intents
        self.has_descriptions = self.validate_descriptions()

    @property
    def multilabel(self) -> bool:
        """Checks if the dataset is multilabel."""
        split = Split.TRAIN if Split.TRAIN in self else f"{Split.TRAIN}_0"
        return isinstance(self[split].features[self.label_feature], Sequence)

    @cached_property
    def n_classes(self) -> int:
        """Returns the number of classes in the dataset."""
        return len(self.intents)

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> "Dataset":
        """Creates a dataset from a dictionary mapping.

        Args:
            mapping: A dictionary representation of the dataset.
        """
        from ._reader import DictReader

        return DictReader().read(mapping)

    @classmethod
    def from_json(cls, filepath: str | Path) -> "Dataset":
        """Loads a dataset from a JSON file.

        Args:
            filepath: Path to the JSON file.
        """
        from ._reader import JsonReader

        return JsonReader().read(filepath)

    @classmethod
    def from_hub(
        cls, repo_name: str, data_split: str = "default", intent_subset_name: str = Split.INTENTS
    ) -> "Dataset":
        """Loads a dataset from the Hugging Face Hub.

        Args:
            repo_name: The name of the Hugging Face repository, like `DeepPavlov/clinc150`.
            data_split: The name of the dataset split to load, defaults to `default`.
            intent_subset_name: The name of the intent subset to load, defaults to `intents`.
        """
        from ._reader import DictReader

        splits = load_dataset(repo_name, data_split)
        mapping = dict(**splits)
        if intent_subset_name in get_dataset_config_names(repo_name):
            mapping[Split.INTENTS] = load_dataset(repo_name, name=intent_subset_name, split=Split.INTENTS).to_list()

        return DictReader().read(mapping)

    def to_multilabel(self) -> "Dataset":
        """Converts dataset labels to multilabel format."""
        for split_name, split in self.items():
            self[split_name] = split.map(self._to_multilabel)
        return self

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Converts the dataset into a dictionary format.

        Returns a dictionary where the keys are dataset splits and the values are lists of samples.
        """
        mapping = {split_name: split.to_list() for split_name, split in self.items()}
        mapping[Split.INTENTS] = [intent.model_dump() for intent in self.intents]
        return mapping

    def to_json(self, filepath: str | Path) -> None:
        """Saves the dataset to a JSON file.

        Args:
            filepath: The file path where the dataset should be saved.
        """
        path = Path(filepath)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4, ensure_ascii=False)

    def push_to_hub(self, repo_name: str, private: bool = False) -> None:
        """Uploads the dataset to the Hugging Face Hub.

        Args:
            repo_name: The ID of the Hugging Face repository.
            private: Whether to make the repository private.
        """
        for split_name, split in self.items():
            split.push_to_hub(repo_name, split=split_name, private=private)

        if self.intents:
            intents = HFDataset.from_list([intent.model_dump() for intent in self.intents])
            intents.push_to_hub(repo_name, config_name=Split.INTENTS, split=Split.INTENTS)

    def get_tags(self) -> list[Tag]:
        """Extracts unique tags from the dataset's intents."""
        tag_mapping = defaultdict(list)
        for intent in self.intents:
            for tag in intent.tags:
                tag_mapping[tag].append(intent.id)
        return [Tag(name=tag, intent_ids=intent_ids) for tag, intent_ids in tag_mapping.items()]

    def get_n_classes(self, split: str) -> int:
        """Calculates the number of unique classes in a dataset split.

        Args:
            split: The dataset split to analyze.
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
        """Converts a sample's label to multilabel format.

        Args:
            sample: A sample from the dataset.
        """
        if isinstance(sample["label"], int):
            ohe_vector = [0] * self.n_classes
            ohe_vector[sample["label"]] = 1
            sample["label"] = ohe_vector
        return sample

    def validate_descriptions(self) -> bool:
        """Validates whether all intents in the dataset contain descriptions."""
        has_any = any(intent.description is not None for intent in self.intents)
        has_all = all(intent.description is not None for intent in self.intents)

        if has_any and not has_all:
            msg = "Some intents have text descriptions, but some do not."
            logger.warning(msg)

        return has_all
