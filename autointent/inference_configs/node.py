from dataclasses import dataclass

from .modules import SearchSpace


@dataclass
class InferenceNodeConfig:
    node_type: str
    module_type: str
    module_config: dict
    load_path: str


@dataclass
class OptimizationNodeConfig:
    node_type: str
    search_space: SearchSpace
    metric: str
