from enum import Enum
from typing import Literal

TASK_TYPES = Literal["multiclass", "multilabel", "multiclass_as_multilabel"]


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


WEIGHT_TYPES = Literal["uniform", "distance", "closest"]

LABEL_TYPE = int | list[int]


class NodeType(str, Enum):
    retrieval = "retrieval"
    prediction = "prediction"
    scoring = "scoring"
    regexp = "regexp"


NODE_TYPES = [NodeType.retrieval, NodeType.prediction, NodeType.scoring, NodeType.regexp]
