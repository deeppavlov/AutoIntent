"""Configuration for the nodes."""

from dataclasses import dataclass
from typing import Any

from autointent.custom_types import NodeType

from ._transformers import CrossEncoderConfig, EmbedderConfig


@dataclass
class InferenceNodeConfig:
    """Configuration for the inference node."""

    node_type: NodeType
    """Type of the node. Should be one of the NODE_TYPES"""
    module_name: str
    """Type of the module. Should be one of the Module"""
    module_config: dict[str, Any]
    """Configuration of the module"""
    load_path: str | None = None
    """Path to the module dump. If None, the module will be trained from scratch"""
    embedder_config: EmbedderConfig | None = None
    """One can override presaved embedder config while loading from file system."""
    cross_encoder_config: CrossEncoderConfig | None = None
    """One can override presaved cross encoder config while loading from file system."""
