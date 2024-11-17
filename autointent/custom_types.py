"""Custom types for AutoIntent."""

from enum import Enum
from typing import Literal, TypedDict

TASK_TYPES = Literal["multiclass", "multilabel", "multiclass_as_multilabel"]


class LogLevel(Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


WEIGHT_TYPES = Literal["uniform", "distance", "closest"]

LabelType = int | list[int]


class BaseMetadataDict(TypedDict):
    """Base metadata dictionary."""


class NodeType(str, Enum):
    """Node types."""

    retrieval = "retrieval"
    prediction = "prediction"
    scoring = "scoring"
    regexp = "regexp"


NODE_TYPES = [NodeType.retrieval, NodeType.prediction, NodeType.scoring, NodeType.regexp]
NodeTypeType = Literal["retrieval", "prediction", "scoring", "regexp"]
