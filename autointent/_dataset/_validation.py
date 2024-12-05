"""File with definitions of DatasetReader and DatasetValidator."""

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from autointent.schemas import Intent, Sample


class DatasetReader(BaseModel):
    """
    A class to represent a dataset reader for handling training, validation, and test data.

    :param train: List of samples for training.
    :param validation: List of samples for validation. Defaults to an empty list.
    :param test: List of samples for testing. Defaults to an empty list.
    :param intents: List of intents associated with the dataset.
    """

    train: list[Sample]
    test: list[Sample] = []
    intents: list[Intent] = []

    @model_validator(mode="after")
    def validate_dataset(self) -> Self:
        """
        Validate the dataset by ensuring intents and data splits are consistent.

        :raises ValueError: If intents or samples are not properly validated.
        :return: The validated DatasetReader instance.
        """
        self._validate_intents()
        for split in [self.train, self.test]:
            self._validate_split(split)
        return self

    def _validate_intents(self) -> Self:
        """
        Validate the intents by checking their IDs for sequential order.

        :raises ValueError: If intent IDs are not sequential starting from 0.
        :return: The DatasetReader instance after validation.
        """
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
        """
        Validate a dataset split to ensure all sample labels reference valid intent IDs.

        :param split: List of samples in a dataset split (train, validation, or test).
        :raises ValueError: If a sample references an invalid or non-existent intent ID.
        :return: The DatasetReader instance after validation.
        """
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
    """A utility class for validating a DatasetReader instance."""

    @staticmethod
    def validate(dataset_reader: DatasetReader) -> DatasetReader:
        """
        Validate a DatasetReader instance.

        :param dataset_reader: An instance of DatasetReader to validate.
        :return: The validated DatasetReader instance.
        """
        return dataset_reader
