import gc
from typing import Any

import torch
from hydra.utils import instantiate

from autointent.configs.node import InferenceNodeConfig
from autointent.nodes.nodes_info import NODES_INFO


class InferenceNode:
    def __init__(self, node_type: str, module_type: str, module_config: dict[str, Any], load_path: str) -> None:
        self.node_info = NODES_INFO[node_type]
        self.module = self.node_info.modules_available[module_type](**module_config)
        self.module.load(load_path)

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> "InferenceNode":
        return instantiate(InferenceNodeConfig, **config)  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        self.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
