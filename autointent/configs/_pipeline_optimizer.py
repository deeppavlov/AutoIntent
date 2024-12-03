"""Pipeline configuration."""

from dataclasses import dataclass

from omegaconf import MISSING

from ._node import NodeOptimizerConfig


@dataclass
class PipelineOptimizerConfig:
    """Configuration for the pipeline optimizer."""

    nodes: list[NodeOptimizerConfig] = MISSING
    """List of the nodes to optimize"""
    _target_: str = "autointent.pipeline.PipelineOptimizer"
