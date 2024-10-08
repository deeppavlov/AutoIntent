from dataclasses import dataclass

from omegaconf import MISSING

from .node import NodeOptimizerConfig


@dataclass
class PipelineOptimizationConfig:
    nodes: list[NodeOptimizerConfig] = MISSING
    _target_: str = "autointent.pipeline.pipeline.Pipeline"
