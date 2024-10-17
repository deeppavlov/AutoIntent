from dataclasses import dataclass

from omegaconf import MISSING

from .node import NodeOptimizerConfig


@dataclass
class PipelineOptimizerConfig:
    nodes: list[NodeOptimizerConfig] = MISSING
    _target_: str = "autointent.pipeline.PipelineOptimizer"
