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
WeightType = Literal["uniform", "distance", "closest"]
"""
Represents weight calculation methods

- "uniform": Equal weight for all elements.
- "distance": Weights based on distance.
- "closest": Prioritizes closest elements.
"""

# Type alias for label representation
SimpleLabel = int
"""Integer label for single-label classification problems."""

MultiLabel = list[int]
"""List label for multi-label classification problems."""

SimpleLabelWithOOS = SimpleLabel | None
"""Integer label for single-label classification problems with OOS samples."""

MultiLabelWithOOS = MultiLabel | None
"""List label for multi-label classification problems with OOS samples."""

ListOfLabels = list[SimpleLabel] | list[MultiLabel]
"""List of labels without OOS-samples that AutoIntent modules can handle."""

ListOfLabelsWithOOS = list[SimpleLabelWithOOS] | list[MultiLabelWithOOS]
"""List of labels with OOS-samples that AutoIntent modules can handle."""

LabelType: TypeAlias = SimpleLabel | MultiLabel
"""
Type alias for label representation

- `int`: For single-label classification.
- `list[int]`: For multi-label classification.
"""

LabelWithOOS = LabelType | None
"""Any label that autointent modules can handle."""

ListOfGenericLabels = ListOfLabels | ListOfLabelsWithOOS
"""List of labels that AutoIntent modules can handle."""


class NodeType(str, Enum):
    """Enumeration of node types in the AutoIntent pipeline."""

    regex = "regex"
    embedding = "embedding"
    scoring = "scoring"
    decision = "decision"


class Split:
    """Enumeration of data splits in the AutoIntent framework.

    Attributes:
        TRAIN: Represents the training data split.
        VALIDATION: Represents the validation data split.
        TEST: Represents the test data split.
        INTENTS: Represents the intents data split.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    INTENTS = "intents"


SamplerType = Literal["tpe", "random"]
"""Hyperparameter tuning strategies:

- `tpe`: :py:class:`optuna.samplers.TPESampler`
- `random`: :py:class:`optuna.samplers.RandomSampler`
"""

ValidationScheme = Literal["ho", "cv"]
"""Validation scheme used in hyperparameter tuning:

- `ho`: hold-out validation
- `cv`: cross-validation
"""


FloatFromZeroToOne = Annotated[float, Interval(ge=0, le=1)]
"""Float value between 0 and 1, inclusive."""

SearchSpaceValidationMode = Literal["raise", "warning", "filter"]
"""Behavior when meet a data-incompatible module in search space:

- `raise`: raise an error
- `warning`: warn user
- `filter`: drop incompatible modules from search space
"""

SearchSpacePreset = Literal[
    "classic-heavy",
    "classic-light",
    "classic-medium",
    "nn-heavy",
    "nn-medium",
    "transformers-heavy",
    "transformers-light",
    "transformers-no-hpo",
    "zero-shot-llm",
    "zero-shot-encoders",
]
"""Some presets that our library supports."""
