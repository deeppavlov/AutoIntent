from dataclasses import dataclass

from omegaconf import MISSING

from .node import InferenceNodeConfig


@dataclass
class InferencePipelineConfig:
    nodes: list[InferenceNodeConfig] = MISSING
    _target_: str = "autointent.pipeline.InferencePipeline"
