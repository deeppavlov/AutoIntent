from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class InferenceNodeConfig:
    node_type: str = MISSING
    module_type: str = MISSING
    module_config: dict = MISSING
    load_path: str = MISSING
    _target_: str = "autointent.nodes.InferenceNode"


@dataclass
class NodeOptimizerConfig:
    node_type: str = MISSING
    search_space: list[dict[str, Any]] = MISSING
    metric: str = MISSING
    _target_: str = "autointent.nodes.NodeOptimizer"
