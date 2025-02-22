"""Dataclasses for the configuration of the :class:`autointent.Embedder` and other objects."""

from ._inference_node import InferenceNodeConfig
from ._optimization import (
    DataConfig,
    LoggingConfig,
    TaskConfig,
    VectorIndexConfig,
)
from ._transformers import CrossEncoderConfig, EmbedderConfig

__all__ = [
    "CrossEncoderConfig",
    "DataConfig",
    "EmbedderConfig",
    "InferenceNodeConfig",
    "InferenceNodeConfig",
    "LoggingConfig",
    "TaskConfig",
    "VectorIndexConfig",
]
