
from pydantic import BaseModel

from autointent.custom_types import LabelType


class Sample(BaseModel):
    utterance: str
    label: LabelType | None = None


class Intent(BaseModel):
    id: int
    name: str | None = None
    tags: list[str] = []
    regexp_full_match: list[str] = []
    regexp_partial_match: list[str] = []
    description: str | None = None


class DatasetReader(BaseModel):
    train: list[Sample]
    validation: list[Sample] = []
    test: list[Sample] = []
    intents: list[Intent] = []


class DatasetValidator:
    @staticmethod
    def validate(dataset_reader: DatasetReader) -> DatasetReader:
        return dataset_reader