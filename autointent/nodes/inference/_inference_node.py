"""InferenceNode class for inference nodes."""

import gc

import torch
from typing_extensions import Self

from autointent.configs import InferenceNodeConfig
from autointent.custom_types import NodeTypeType
from autointent.modules import Module
from autointent.nodes.nodes_info import NODES_INFO


class InferenceNode:
    """Inference node class."""

    def __init__(self, module: Module, node_type: NodeTypeType) -> None:
        """
        Initialize the inference node.

        :param module: Module to use for inference
        :param node_type: Node types
        """
        self.module = module
        self.node_type = node_type

    @classmethod
    def from_config(cls, config: InferenceNodeConfig) -> Self:
        """
        Initialize from config.

        :param config: Configuration for the node.
        """
        node_info = NODES_INFO[config.node_type]
        module = node_info.modules_available[config.module_type](**config.module_config)
        if config.load_path is not None:
            module.load(config.load_path)
        return cls(module, config.node_type)

    def clear_cache(self) -> None:
        """Clear cache."""
        self.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
