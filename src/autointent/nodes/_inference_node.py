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
        module_cls = node_info.modules_available[config.module_name]
        module = module_cls.load(
            config.load_path,
            embedder_config=getattr(config, "embedder_config", None),
            cross_encoder_config=getattr(config, "cross_encoder_config", None),
        )
        return cls(module, config.node_type)

    def clear_cache(self) -> None:
        """Clear cache."""
        self.module.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
