from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class InferenceNodeConfig:
    node_type: str
    module_type: str
    module_config: dict
    load_path: str


@dataclass
class NodeOptimizerConfig:
    node_type: str = MISSING
    search_space: list[dict] = MISSING
    metric: str = MISSING
    _target_: str = "autointent.nodes.NodeOptimizer"
