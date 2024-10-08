from dataclasses import dataclass

from autointent.nodes import NodeInfo

from .modules import ModuleConfig, SearchSpaceDataclass


@dataclass
class InferenceNodeConfig:
    node_type: str
    module_type: str
    module_config: ModuleConfig
    load_path: str
    _target_: str = "autointent.nodes.InferenceNode"


@dataclass
class NodeOptimizerConfig:
    node_info: NodeInfo
    search_space: list[SearchSpaceDataclass]
    metric: str
    _target_: str = "autointent.nodes.NodeOptimizer"
