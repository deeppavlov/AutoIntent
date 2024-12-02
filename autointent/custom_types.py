"""Custom types for AutoIntent.

This module defines custom types, enumerations, and constants used throughout
the AutoIntent framework for improved type safety and clarity.
"""

from enum import Enum
from typing import Literal, TypedDict

# Literal type for task types
TASK_TYPES = Literal["multiclass", "multilabel", "multiclass_as_multilabel"]
"""
Represents the supported task types in AutoIntent:
- "multiclass": Single-label classification.
- "multilabel": Multi-label classification.
- "multiclass_as_multilabel": Multi-class treated as multi-label.
"""


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
Represents weight calculation methods:
- "uniform": Equal weight for all elements.
- "distance": Weights based on distance.
- "closest": Prioritizes closest elements.
"""

# Type alias for label representation
LabelType = int | list[int]
"""
Type alias for label representation:
- `int`: For single-label classification.
- `list[int]`: For multi-label classification.
"""


class BaseMetadataDict(TypedDict):
    """Base metadata dictionary for storing additional information."""


class NodeType(str, Enum):
    """Enumeration of node types in the AutoIntent pipeline."""

    regexp = "regexp"
    retrieval = "retrieval"
    scoring = "scoring"
    prediction = "prediction"
