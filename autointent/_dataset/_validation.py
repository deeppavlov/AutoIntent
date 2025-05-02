"""File containing definitions of DatasetReader and DatasetValidator for handling dataset operations."""

from pydantic import BaseModel, ConfigDict, model_validator

from autointent.schemas import Intent, Sample


class DatasetReader(BaseModel):
    """Represents a dataset reader for handling training, validation, and test data splits.

    Attributes:
        train: List of samples for training. Defaults to an empty list.
        train_0: List of samples for scoring module training. Defaults to an empty list.
        train_1: List of samples for decision module training. Defaults to an empty list.
        validation: List of samples for validation. Defaults to an empty list.
        validation_0: List of samples for scoring module validation. Defaults to an empty list.
        validation_1: List of samples for decision module validation. Defaults to an empty list.
        test: List of samples for testing. Defaults to an empty list.
        intents: List of intents associated with the dataset.
    """

    train: list[Sample] = []
    train_0: list[Sample] = []
    train_1: list[Sample] = []
    validation: list[Sample] = []
    validation_0: list[Sample] = []
    validation_1: list[Sample] = []
    test: list[Sample] = []
    intents: list[Intent] = []

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_dataset(self) -> "DatasetReader":
        """Validates dataset integrity by ensuring consistent data splits and intent mappings.

        Raises:
            ValueError: If data splits are inconsistent or intent mappings are incorrect.

        Returns:
            DatasetReader: The validated dataset reader instance.
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
            self.train,
            self.train_0,
            self.train_1,
            self.validation,
            self.validation_0,
            self.validation_1,
            self.test,
        ]
        splits = [split for split in splits if split]

        n_classes = self._validate_classes(splits)

        self._validate_intents(n_classes)

        for split in splits:
            self._validate_split(split)
        return self

    def _get_n_classes(self, split: list[Sample]) -> int:
        """Determines the number of unique classes in a dataset split.

        Args:
            split (list[Sample]): List of samples in a dataset split.

        Returns:
            int: The number of unique classes.
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

    def _validate_classes(self, splits: list[list[Sample]]) -> int:
        """Ensures that all dataset splits have the same number of classes.

        Args:
            splits (list[list[Sample]]): List of dataset splits.

        Raises:
            ValueError: If the number of classes is inconsistent across splits or if no classes are found.

        Returns:
            int: The number of unique classes.
        """
        n_classes = [self._get_n_classes(split) for split in splits]
        if len(set(n_classes)) != 1:
            message = (
                f"Mismatch in number of classes across splits. Found class counts: {n_classes}. "
                "Ensure all splits have the same number of classes."
            )
            raise ValueError(message)
        if not n_classes[0]:
            message = "Number of classes is zero or undefined. Ensure at least one class is present in the splits."
            raise ValueError(message)
        return n_classes[0]

    def _validate_intents(self, n_classes: int) -> "DatasetReader":
        """Ensures intent IDs are sequential and match the number of classes.

        Args:
            n_classes (int): The expected number of classes based on dataset splits.

        Raises:
            ValueError: If intent IDs are not sequential or valid.

        Returns:
            DatasetReader: The validated dataset reader instance.
        """
        if not self.intents:
            self.intents = [Intent(id=idx) for idx in range(n_classes)]
            return self
        self.intents = sorted(self.intents, key=lambda intent: intent.id)
        intent_ids = [intent.id for intent in self.intents]
        if intent_ids != list(range(len(self.intents))):
            message = (
                f"Invalid intent IDs. Expected sequential IDs from 0 to {len(self.intents) - 1}, but got {intent_ids}."
            )
            raise ValueError(message)
        return self

    def _validate_split(self, split: list[Sample]) -> "DatasetReader":
        """Validate a dataset split to ensure all sample labels reference valid intent IDs.

        Args:
            split: List of samples in a dataset split.

        Raises:
            ValueError: If a sample references an invalid or non-existent intent ID.

        Returns:
            DatasetReader: The validated dataset reader instance.
        """
        intent_ids = {intent.id for intent in self.intents}
        for sample in split:
            message = (
                f"Sample with label {sample.label} and utterance {sample.utterance[:10]}... "
                f"references a non-existent intent ID. Valid IDs are {intent_ids}."
            )
            if isinstance(sample.label, int) and sample.label not in intent_ids:
                raise ValueError(message)
            if isinstance(sample.label, list) and any(label not in intent_ids for label in sample.label):
                raise ValueError(message)
        return self


class DatasetValidator:
    """Utility class for validating a DatasetReader instance."""

    @staticmethod
    def validate(dataset_reader: DatasetReader) -> DatasetReader:
        """Validates a DatasetReader instance.

        Args:
            dataset_reader (DatasetReader): The dataset reader instance to validate.

        Returns:
            DatasetReader: The validated dataset reader instance.
        """
        return dataset_reader
