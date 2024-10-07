from dataclasses import dataclass

from autointent.nodes import NodeInfo

from .modules import ModuleConfig, SearchSpace


@dataclass
class InferenceNodeConfig:
    _target_: str = "autointent.nodes.InferenceNode"
    node_type: str
    module_type: str
    module_config: ModuleConfig
    load_path: str


@dataclass
class NodeOptimizerConfig:
    _target_: str = "autointent.nodes.NodeOptimizer"
    node_info: NodeInfo
    search_space: list[SearchSpace]
    metric: str
