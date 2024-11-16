from dataclasses import dataclass

from omegaconf import MISSING

from .node import InferenceNodeConfig


@dataclass
class InferencePipelineConfig:
    """Configuration for the inference pipeline."""

    nodes: list[InferenceNodeConfig] = MISSING
    """List of nodes in the inference pipeline"""
    _target_: str = "autointent.pipeline.InferencePipeline"
