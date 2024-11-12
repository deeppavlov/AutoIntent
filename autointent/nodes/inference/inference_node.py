import gc

import torch
from typing_extensions import Self

from autointent.configs.node import InferenceNodeConfig
from autointent.modules.base import Module
from autointent.nodes.nodes_info import NODES_INFO


class InferenceNode:
    def __init__(self, module: Module, node_type: str) -> None:
        self.module = module
        self.node_type = node_type

    @classmethod
    def from_config(cls, config: InferenceNodeConfig) -> Self:
        node_info = NODES_INFO[config.node_type]
        module = node_info.modules_available[config.module_type](**config.module_config)
        if config.load_path is not None:
            module.load(config.load_path)
        return cls(module, config.node_type)

    def clear_cache(self) -> None:
        self.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
