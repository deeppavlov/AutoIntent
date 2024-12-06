"""Configuration for the nodes."""

from dataclasses import dataclass
from typing import Any

from autointent.custom_types import NodeType


@dataclass
class InferenceNodeConfig:
    """Configuration for the inference node."""

    node_type: NodeType
    """Type of the node. Should be one of the NODE_TYPES"""
    module_type: str
    """Type of the module. Should be one of the Module"""
    module_config: dict[str, Any]
    """Configuration of the module"""
    load_path: str | None = None
    """Path to the module dump. If None, the module will be trained from scratch"""
