from enum import Enum


class ClassificationMode(Enum):
    multiclass = "multiclass"
    multilabel = "multilabel"
    multiclass_as_multilabel = "multiclass_as_multilabel"


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
