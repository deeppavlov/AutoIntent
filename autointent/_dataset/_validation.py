"""File with definitions of DatasetReader and DatasetValidator."""

from pydantic import BaseModel, model_validator

from autointent.schemas import Intent, Sample


class DatasetReader(BaseModel):
    """
    A class to represent a dataset reader for handling training, validation, and test data.

    :param train: List of samples for training. Defaults to an empty list.
    :param train_0: List of samples for scoring module training. Defaults to an empty list.
    :param train_1: List of samples for decision module training. Defaults to an empty list.
    :param validation: List of samples for validation. Defaults to an empty list.
    :param validation_0: List of samples for scoring module validation. Defaults to an empty list.
    :param validation_1: List of samples for decision module validation. Defaults to an empty list.
    :param test: List of samples for testing. Defaults to an empty list.
    :param intents: List of intents associated with the dataset.
    """

    train: list[Sample] = []
    train_0: list[Sample] = []
    train_1: list[Sample] = []
    validation: list[Sample] = []
    validation_0: list[Sample] = []
    validation_1: list[Sample] = []
    test: list[Sample] = []
    intents: list[Intent] = []

    @model_validator(mode="after")
    def validate_dataset(self) -> "DatasetReader":
        """
        Validate the dataset by ensuring intents and data splits are consistent.

        :raises ValueError: If intents or samples are not properly validated.
        :return: The validated DatasetReader instance.
        """
        if self.train and (self.train_0 or self.train_1):
            message = "If `train` is provided, `train_0` and `train_1` should be empty."
            raise ValueError(message)
        if not self.train and (not self.train_0 or not self.train_1):
            message = "Both `train_0` and `train_1` must be provided if `train` is empty."
            raise ValueError(message)

        if self.validation and (self.validation_0 or self.validation_1):
            message = "If `validation` is provided, `validation_0` and `validation_1` should be empty."
            raise ValueError(message)
        if not self.validation:
            message = "Either both `validation_0` and `validation_1` must be provided, or neither of them."
            if not self.validation_0 and self.validation_1:
                raise ValueError(message)
            if self.validation_0 and not self.validation_1:
                raise ValueError(message)

        splits = [
            self.train, self.train_0, self.train_1, self.validation, self.validation_0, self.validation_1, self.test,
        ]
        splits = [split for split in splits if split]

        n_classes = [self._get_n_classes(split) for split in splits]
        if len(set(n_classes)) != 1:
            message = (
                f"Mismatch in number of classes across splits. Found class counts: {n_classes}. "
                "Ensure all splits have the same number of classes."
            )
            raise ValueError(message)
        if not n_classes[0]:
            message = (
                "Number of classes is zero or undefined. "
                "Ensure at least one class is present in the splits."
            )
            raise ValueError(message)

        self._validate_intents(n_classes[0])

        for split in splits:
            self._validate_split(split)
        return self

    def _get_n_classes(self, split: list[Sample]) -> int:
        """
        Get the number of classes in a dataset split.

        :param split: List of samples in a dataset split (train, validation, or test).
        :return: The number of classes.
        """
        classes = set()
        for sample in split:
            match sample.label:
                case int():
                    classes.add(sample.label)
                case list():
                    for label in sample.label:
                        classes.add(label)
        return len(classes)

    def _validate_intents(self, n_classes: int) -> "DatasetReader":
        """
        Validate the intents by checking their IDs for sequential order.

        :param n_classes: The number of classes in the dataset.
        :raises ValueError: If intent IDs are not sequential starting from 0.
        :return: The DatasetReader instance after validation.
        """
        if not self.intents:
            self.intents = [Intent(id=idx) for idx in range(n_classes)]
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

    def _validate_split(self, split: list[Sample]) -> "DatasetReader":
        """
        Validate a dataset split to ensure all sample labels reference valid intent IDs.

        :param split: List of samples in a dataset split (train, validation, or test).
        :raises ValueError: If a sample references an invalid or non-existent intent ID.
        :return: The DatasetReader instance after validation.
        """
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
