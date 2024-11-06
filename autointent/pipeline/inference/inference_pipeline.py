from typing import Any

from hydra.utils import instantiate

from autointent.configs.inference_pipeline import InferencePipelineConfig
from autointent.custom_types import LabelType, NodeType
from autointent.configs.node import InferenceNodeConfig
from autointent.context import Context
from autointent.custom_types import LabelType
from autointent.nodes.inference import InferenceNode


class InferencePipeline:
    def __init__(self, nodes: list[InferenceNode]) -> None:
        self.nodes = {node.node_info.node_type: node for node in nodes}

    @classmethod
    def from_config(cls, nodes_configs: list[InferenceNodeConfig]) -> "InferencePipeline":
        nodes = [InferenceNode.from_config(cfg) for cfg in nodes_configs]
        return cls(nodes)

    def predict(self, utterances: list[str]) -> list[LabelType]:
        scores = self.nodes[NodeType.scoring].module.predict(utterances)
        return self.nodes[NodeType.prediction].module.predict(scores)  # type: ignore[return-value]

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        pass

    @classmethod
    def from_context(cls, context: Context) -> "InferencePipeline":
        if not context.has_saved_modules():
            config = context.optimization_info.get_inference_nodes_config()
            return cls.from_config(config)
        nodes = [
            InferenceNode(module, node_type)
            for node_type, module in context.optimization_info.get_best_modules().items()
        ]
        return InferencePipeline(nodes)
