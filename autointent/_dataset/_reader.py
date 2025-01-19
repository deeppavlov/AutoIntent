"""Base classes and implementations for reading datasets in various formats."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset

from ._dataset import Dataset
from ._validation import DatasetReader, DatasetValidator


class BaseReader(ABC):
    """
    Abstract base class for dataset readers. Defines the interface for reading datasets.

    Subclasses must implement the `_read` method to specify how the dataset is read.

    :raises NotImplementedError: If `_read` is not implemented by the subclass.
    """

    def read(self, *args: Any, **kwargs: Any) -> Dataset:  # noqa: ANN401
        """
        Read and validate the dataset, converting it to the standard `Dataset` format.

        :param args: Positional arguments for the `_read` method.
        :param kwargs: Keyword arguments for the `_read` method.
        :return: A `Dataset` object containing the dataset splits and intents.
        """
        dataset_reader = DatasetValidator.validate(self._read(*args, **kwargs))
        splits = dataset_reader.model_dump(exclude={"intents"}, exclude_defaults=True)
        return Dataset(
            {split_name: HFDataset.from_list(split) for split_name, split in splits.items()},
            intents=sorted(dataset_reader.intents, key=lambda intent: intent.id),
        )

    @abstractmethod
    def _read(self, *args: Any, **kwargs: Any) -> DatasetReader:  # noqa: ANN401
        """
        Abstract method for reading a dataset.

        This must be implemented by subclasses to provide specific reading logic.

        :param args: Positional arguments for dataset reading.
        :param kwargs: Keyword arguments for dataset reading.
        :return: A `DatasetReader` instance representing the dataset.
        """
        ...


class DictReader(BaseReader):
    """Dataset reader that processes datasets provided as Python dictionaries."""

    def _read(self, mapping: dict[str, Any]) -> DatasetReader:
        """
        Read a dataset from a dictionary and validate it.

        :param mapping: A dictionary representing the dataset.
        :return: A validated `DatasetReader` instance.
        """
        return DatasetReader.model_validate(mapping)


class JsonReader(BaseReader):
    """Dataset reader that processes datasets from JSON files."""

    def _read(self, filepath: str | Path) -> DatasetReader:
        """
        Read a dataset from a JSON file and validate it.

        :param filepath: Path to the JSON file containing the dataset.
        :type filepath: str or Path
        :return: A validated `DatasetReader` instance.
        """
        with Path(filepath).open() as file:
            return DatasetReader.model_validate(json.load(file))
