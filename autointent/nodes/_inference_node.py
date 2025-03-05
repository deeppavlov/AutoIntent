"""InferenceNode class for inference nodes."""

import gc

import torch

from autointent.configs import InferenceNodeConfig
from autointent.custom_types import NodeType
from autointent.modules.base import BaseModule
from autointent.nodes.info import NODES_INFO


class InferenceNode:
    """Inference node class."""

    def __init__(self, module: BaseModule, node_type: NodeType) -> None:
        """Initialize the inference node.

        Args:
            module: Module to use for inference
            node_type: Node types
        """
        self.module = module
        self.node_type = node_type

    @classmethod
    def from_config(cls, config: InferenceNodeConfig) -> "InferenceNode":
        """Initialize from config.

        Args:
            config: Config to init from
        """
        node_info = NODES_INFO[config.node_type]
        module = node_info.modules_available[config.module_name](**config.module_config)
        module.load(
            config.load_path,
            embedder_config=config.embedder_config,
            cross_encoder_config=config.cross_encoder_config,
        )
        return cls(module, config.node_type)

    def clear_cache(self) -> None:
        """Clear cache."""
        self.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
