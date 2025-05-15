"""Dataclasses for the configuration of the :class:`autointent.Embedder` and other objects."""

from ._inference_node import InferenceNodeConfig
from ._optimization import DataConfig, LoggingConfig
from ._transformers import CNNConfig, CrossEncoderConfig, EmbedderConfig, HFModelConfig, TaskTypeEnum, TokenizerConfig

__all__ = [
    "CNNConfig",
    "CrossEncoderConfig",
    "DataConfig",
    "EmbedderConfig",
    "HFModelConfig",
    "InferenceNodeConfig",
    "LoggingConfig",
    "TaskTypeEnum",
    "TokenizerConfig"
]
