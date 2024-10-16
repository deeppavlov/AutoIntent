from typing import Any, TypeVar

from hydra.utils import instantiate

from autointent.configs.inference_pipeline import InferencePipelineConfig
from autointent.nodes.inference import InferenceNode

PipelineType = TypeVar("PipelineType", bound="InferencePipeline")


class InferencePipeline:
    def __init__(self, nodes: list[InferenceNode]) -> None:
        self.nodes = {node.node_info.node_type: node for node in nodes}

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> PipelineType:
        return instantiate(InferencePipelineConfig, **config)

    def predict(self, utterances: list[str]) -> list[int] | list[list[int]]:
        scores = self.nodes["scoring"].module.predict(utterances)
        return self.nodes["prediction"].module.predict(scores)
