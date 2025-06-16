"""Base classes and implementations for reading datasets in various formats."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset

from ._dataset import Dataset
from ._validation import DatasetReader, DatasetValidator


class BaseReader(ABC):
    """Abstract base class for dataset readers.

    This class defines the interface for reading datasets from various sources.
    Subclasses must implement the `_read` method to specify how a dataset should
    be read and processed.

    Raises:
        NotImplementedError: If `_read` is not implemented in a subclass.
    """

    def read(self, *args: Any, **kwargs: Any) -> Dataset:  # noqa: ANN401
        """Reads and validates the dataset, converting it into a standardized `Dataset` object.

        This method first calls the `_read` method (implemented by subclasses)
        to retrieve the dataset, then validates it using `DatasetValidator`.
        The validated dataset is converted into the standard `Dataset` format.

        Args:
            *args: Positional arguments passed to the `_read` method.
            **kwargs: Keyword arguments passed to the `_read` method.

        Returns:
            Dataset: A standardized dataset object containing the dataset splits and intents.
        """
        dataset_reader = DatasetValidator.validate(self._read(*args, **kwargs))
        splits = dataset_reader.model_dump(exclude={"intents"}, exclude_defaults=True)
        return Dataset(
            {split_name: HFDataset.from_list(split) for split_name, split in splits.items()},
            intents=sorted(dataset_reader.intents, key=lambda intent: intent.id),
        )

    @abstractmethod
    def _read(self, *args: Any, **kwargs: Any) -> DatasetReader:  # noqa: ANN401
        """Abstract method for reading a dataset.

        This method must be implemented by subclasses to define the specific logic
        for reading datasets from different sources (e.g., dictionaries, JSON files).

        Args:
            *args: Positional arguments for dataset reading.
            **kwargs: Keyword arguments for dataset reading.

        Returns:
            DatasetReader: A dataset representation that will be validated and processed.
        """
        ...


class DictReader(BaseReader):
    """Dataset reader that processes datasets provided as Python dictionaries.

    This reader expects datasets in a dictionary format and validates the dataset
    structure before converting it into a standardized `Dataset` object.
    """

    def _read(self, mapping: dict[str, Any]) -> DatasetReader:
        """Reads and validates a dataset from a dictionary.

        Args:
            mapping: A dictionary representing the dataset.

        Returns:
            DatasetReader: A validated dataset representation.
        """
        return DatasetReader.model_validate(mapping)


class JsonReader(BaseReader):
    """Dataset reader that loads and processes datasets from JSON files.

    This reader reads datasets stored as JSON files and validates them before
    converting them into a standardized `Dataset` object.
    """

    def _read(self, filepath: str | Path) -> DatasetReader:
        """Reads and validates a dataset from a JSON file.

        Args:
            filepath: Path to the JSON file containing the dataset.

        Returns:
            DatasetReader: A validated dataset representation.
        """
        with Path(filepath).open(encoding="utf-8") as file:
            return DatasetReader.model_validate(json.load(file))
