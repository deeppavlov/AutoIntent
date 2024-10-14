from autointent.nodes.nodes_info import NODES_INFO


class InferenceNode:
    def __init__(self, node_type: str,  module_type: str,  module_config: dict, load_path: str) -> None:
        self.node_info = NODES_INFO[node_type]
        self.module = self.node_info.modules_available[module_type](**module_config)
        self.load_path = load_path

    def load(self) -> None:
        self.module.load(self.load_path)
