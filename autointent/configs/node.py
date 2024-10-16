from dataclasses import dataclass

from omegaconf import MISSING

from .modules import ModuleConfig


@dataclass
class InferenceNodeConfig:
    node_type: str = MISSING
    module_type: str = MISSING
    module_config: ModuleConfig = MISSING
    load_path: str = MISSING
    _target_: str = "autointent.nodes.InferenceNode"


@dataclass
class NodeOptimizerConfig:
    node_type: str = MISSING
    search_space: list[dict] = MISSING
    metric: str = MISSING
    _target_: str = "autointent.nodes.NodeOptimizer"
