from pydantic import BaseModel, model_validator
from typing_extensions import Self

from .schemas import Intent, Sample


class DatasetReader(BaseModel):
    train: list[Sample]
    validation: list[Sample] = []
    test: list[Sample] = []
    intents: list[Intent] = []

    @model_validator(mode="after")
    def validate_dataset(self) -> Self:
        self._validate_intents()
        for split in [self.train, self.validation, self.test]:
            self._validate_split(split)
        return self

    def _validate_intents(self) -> Self:
        if not self.intents:
            return self
        self.intents = sorted(self.intents, key=lambda intent: intent.id)
        intent_ids = [intent.id for intent in self.intents]
        if intent_ids != list(range(len(self.intents))):
            message = (
                f"Invalid intent IDs. Expected sequential IDs from 0 to {len(self.intents) - 1}, "
                f"but got {intent_ids}."
            )
            raise ValueError(message)
        return self

    def _validate_split(self, split: list[Sample]) -> Self:
        if not split or not self.intents:
            return self
        intent_ids = {intent.id for intent in self.intents}
        for sample in split:
            message = (
                f"Sample with label {sample.label} references a non-existent intent ID. " f"Valid IDs are {intent_ids}."
            )
            if isinstance(sample.label, int) and sample.label not in intent_ids:
                raise ValueError(message)
            if isinstance(sample.label, list) and any(label not in intent_ids for label in sample.label):
                raise ValueError(message)
        return self


class DatasetValidator:
    @staticmethod
    def validate(dataset_reader: DatasetReader) -> DatasetReader:
        return dataset_reader
