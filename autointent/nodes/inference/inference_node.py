from autointent.nodes.base import NodeInfo


class InferenceNode:
    def __init__(self, node_info: NodeInfo):
        self.node_info = node_info

    def load(self): ...
