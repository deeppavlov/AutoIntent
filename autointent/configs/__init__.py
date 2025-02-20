"""Dataclasses for the configuration of the :class:`autointent.Embedder` and other objects."""

from ._inference_node import InferenceNodeConfig
from ._optimization import (
    DataConfig,
    LoggingConfig,
    VectorIndexConfig,
)

__all__ = [
    "DataConfig",
    "InferenceNodeConfig",
    "InferenceNodeConfig",
    "LoggingConfig",
    "VectorIndexConfig",
]
