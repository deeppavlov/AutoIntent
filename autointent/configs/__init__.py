"""Dataclasses for the configuration of the :class:`autointent.Embedder` and other objects."""

from ._inference_node import InferenceNodeConfig
from ._optimization_cli import (
    CrossEncoderConfig,
    DataConfig,
    EmbedderConfig,
    LoggingConfig,
    OptimizationConfig,
    TaskConfig,
    VectorIndexConfig,
)

__all__ = [
    "CrossEncoderConfig",
    "DataConfig",
    "EmbedderConfig",
    "InferenceNodeConfig",
    "InferenceNodeConfig",
    "LoggingConfig",
    "OptimizationConfig",
    "TaskConfig",
    "VectorIndexConfig",
]
