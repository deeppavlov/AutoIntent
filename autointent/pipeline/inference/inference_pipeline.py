from typing import Any, TypeVar

from hydra.utils import instantiate

from autointent.configs.inference_pipeline import InferencePipelineConfig
from autointent.nodes.inference import InferenceNode

PipelineType = TypeVar("PipelineType", bound="InferencePipeline")


class InferencePipeline:
    def __init__(self, nodes: list[InferenceNode]) -> None:
        self.nodes = nodes

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> PipelineType:
        return instantiate(InferencePipelineConfig, **config)
