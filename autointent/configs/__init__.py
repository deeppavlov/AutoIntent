"""Dataclasses for the configuration of the :class:`autointent.Embedder` and other objects."""

from ._inference_node import InferenceNodeConfig
from ._optimization import (
    DataConfig,
    LoggingConfig,
    TaskConfig,
    VectorIndexConfig,
)

__all__ = [
    "DataConfig",
    "InferenceNodeConfig",
    "InferenceNodeConfig",
    "LoggingConfig",
    "TaskConfig",
    "VectorIndexConfig",
]
