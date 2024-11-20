import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset

from .dataset import Dataset
from .validation import DatasetReader, DatasetValidator


class BaseReader(ABC):
    def read(self, *args: Any, **kwargs: Any) -> Dataset:
        dataset_reader = DatasetValidator.validate(self._read(*args, **kwargs))
        splits = dataset_reader.model_dump(exclude={"intents"}, exclude_defaults=True)
        return Dataset(
            {split_name: HFDataset.from_list(split) for split_name, split in splits.items()},
            intents=sorted(dataset_reader.intents, key=lambda intent: intent.id),
        )

    @abstractmethod
    def _read(self, *args: Any, **kwargs: Any) -> DatasetReader:
        ...


class DictReader(BaseReader):
    def _read(self, mapping: dict[str, Any]) -> DatasetReader:
        return DatasetReader.model_validate(mapping)


class JsonReader(BaseReader):
    def _read(self, filepath: str | Path) -> DatasetReader:
        with Path(filepath).open() as file:
            return DatasetReader.model_validate(json.load(file))
