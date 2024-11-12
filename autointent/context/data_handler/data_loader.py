
from datasets import Dataset
from pydantic import BaseModel

from autointent.custom_types import LabelType

from .dataset import DatasetDict


class Utterance(BaseModel):
    text: str
    label: LabelType | None = None


class Intent(BaseModel):
    id: int
    name: str | None = None
    tags: list[str] = []
    regexp_full_match: list[str] = []
    regexp_partial_match: list[str] = []
    description: str | None = None


class DatasetLoader(BaseModel):
    train: list[Utterance]
    validation: list[Utterance] = []
    test: list[Utterance] = []
    intents: list[Intent] = []


    def to_dataset(self) -> DatasetDict:
        return DatasetDict(
            {
                split_name: Dataset.from_list(split)
                for split_name, split in self.model_dump(exclude_defaults=True)
            }
        )
