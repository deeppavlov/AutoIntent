from autointent.configs.node import InferenceNodeConfig
from autointent.context import Context
from autointent.custom_types import LabelType
from autointent.nodes.inference import InferenceNode


class InferencePipeline:
    def __init__(self, nodes: list[InferenceNode]) -> None:
        self.nodes = {n.node_type: n for n in nodes}

    @classmethod
    def from_config(cls, nodes_configs: list[InferenceNodeConfig]) -> None:
        nodes = [InferenceNode.from_config(cfg) for cfg in nodes_configs]
        return cls(nodes)

    def predict(self, utterances: list[str]) -> list[LabelType]:
        scores = self.nodes["scoring"].module.predict(utterances)
        return self.nodes["prediction"].module.predict(scores)  # type: ignore[return-value]

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
