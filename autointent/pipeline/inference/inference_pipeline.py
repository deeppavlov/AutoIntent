from typing import Any

from hydra.utils import instantiate

from autointent.configs.inference_pipeline import InferencePipelineConfig
from autointent.custom_types import LabelType, NodeType
from autointent.nodes.inference import InferenceNode


class InferencePipeline:
    def __init__(self, nodes: list[InferenceNode]) -> None:
        self.nodes = {node.node_info.node_type: node for node in nodes}

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> "InferencePipeline":
        return instantiate(InferencePipelineConfig, **config)  # type: ignore[no-any-return]

    def predict(self, utterances: list[str]) -> list[LabelType]:
        scores = self.nodes[NodeType.scoring].module.predict(utterances)
        return self.nodes[NodeType.prediction].module.predict(scores)  # type: ignore[return-value]

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        pass
