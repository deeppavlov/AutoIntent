
from pydantic import BaseModel

from .schemas import Intent, Sample


class DatasetReader(BaseModel):
    train: list[Sample]
    validation: list[Sample] = []
    test: list[Sample] = []
    intents: list[Intent] = []


class DatasetValidator:
    @staticmethod
    def validate(dataset_reader: DatasetReader) -> DatasetReader:
        return dataset_reader
