from autointent.nodes.nodes_info import NODES_INFO


class InferenceNode:
    def __init__(self, node_type: str):
        self.node_info = NODES_INFO[node_type]

    def load(self): ...
