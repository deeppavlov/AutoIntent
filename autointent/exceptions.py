"""Custom exceptions that are raised by AutoIntent."""


class WrongClassificationError(Exception):
    """Exception raised when a classification module is used with incompatible data.

    This error typically occurs when a multiclass module is called on multilabel data
    or vice versa.

    Args:
        message: Error message, defaults to a standard incompatibility message
    """

    def __init__(self, message: str = "Multiclass module is called on multilabel data or vice-versa") -> None:
        """Initialize the exception.

        Args:
            message: Error message, defaults to a standard incompatibility message
        """
        self.message = message
        super().__init__(message)


class MismatchNumClassesError(Exception):
    """Exception raised when the data contains an incompatible number of classes.

    This error indicates that the number of classes in the input data does not match
    the expected number of classes for the module.

    Args:
        message: Error message, defaults to a standard class incompatibility message
    """

    def __init__(self, message: str | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Error message, defaults to a standard incompatibility message
        """
        self.message = (
            message or "Provided scores number don't match with number of classes which module was trained on."
        )
        super().__init__(message)
