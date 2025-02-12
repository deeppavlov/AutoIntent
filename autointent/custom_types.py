"""Custom types for AutoIntent.

This module defines custom types, enumerations, and constants used throughout
the AutoIntent framework for improved type safety and clarity.
"""

from enum import Enum
from typing import Annotated, Literal, TypeAlias

from annotated_types import Interval


class LogLevel(Enum):
    """Logging levels for the AutoIntent framework."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Literal type for weight types in specific operations
WEIGHT_TYPES = Literal["uniform", "distance", "closest"]
"""
Represents weight calculation methods

- "uniform": Equal weight for all elements.
- "distance": Weights based on distance.
- "closest": Prioritizes closest elements.
"""

# Type alias for label representation
SimpleLabel = int
MultiLabel = list[int]
SimpleLabelWithOOS = SimpleLabel | None
MultiLabelWithOOS = MultiLabel | None
ListOfLabels = list[SimpleLabel] | list[MultiLabel]
ListOfLabelsWithOOS = list[SimpleLabelWithOOS] | list[MultiLabelWithOOS]
LabelType: TypeAlias = SimpleLabel | MultiLabel
LabelWithOOS = LabelType | None
ListOfGenericLabels = ListOfLabels | ListOfLabelsWithOOS
"""
Type alias for label representation

- `int`: For single-label classification.
- `list[int]`: For multi-label classification.
"""


class NodeType(str, Enum):
    """Enumeration of node types in the AutoIntent pipeline."""

    regexp = "regexp"
    embedding = "embedding"
    scoring = "scoring"
    decision = "decision"


class Split:
    """
    Constants representing dataset splits.

    :cvar str TRAIN: Training split.
    :cvar str VALIDATION: Validation split.
    :cvar str TEST: Testing split.
    :cvar str INTENTS: Intents split.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    INTENTS = "intents"


SamplerType = Literal["brute", "tpe", "random"]
ValidationScheme = Literal["ho", "cv"]


FloatFromZeroToOne = Annotated[float, Interval(ge=0, le=1)]
"""Float value between 0 and 1, inclusive."""
