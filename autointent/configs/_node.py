"""Configuration for the nodes."""

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING

from autointent.custom_types import NodeType


@dataclass
class InferenceNodeConfig:
    """Configuration for the inference node."""

    node_type: NodeType = MISSING
    """Type of the node. Should be one of the NODE_TYPES"""
    module_type: str = MISSING  # TODO: add custom type
    """Type of the module. Should be one of the Module"""
    module_config: dict[str, Any] = MISSING
    """Configuration of the module"""
    load_path: str | None = None
    """Path to the module dump. If None, the module will be trained from scratch"""
    _target_: str = "autointent.nodes.InferenceNode"


@dataclass
class NodeOptimizerConfig:
    """Configuration for the node optimizer."""

    node_type: NodeType = MISSING
    """Type of the node. Should be one of the NODE_TYPES"""
    search_space: list[dict[str, Any]] = MISSING
    """Search space for the optimization"""
    metric: str = MISSING  # TODO: add custom type
    """Metric to optimize"""
    _target_: str = "autointent.nodes.NodeOptimizer"
