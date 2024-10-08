from dataclasses import dataclass

from .node import NodeOptimizerConfig


@dataclass
class PipelineOptimizationConfig:
    nodes: list[NodeOptimizerConfig]
    _target_: str = "autointent.pipeline.pipeline.Pipeline"
