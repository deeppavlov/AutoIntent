"""Configuration for the nodes."""

from typing import Any

from autointent.custom_types import NodeType

from ._transformers import CrossEncoderConfig, EmbedderConfig


class InferenceNodeConfig:
    """Configuration for the inference node."""

    def __init__(
        self,
        node_type: NodeType,
        module_name: str,
        module_config: dict[str, Any],
        load_path: str,
        embedder_config: EmbedderConfig | None = None,
        cross_encoder_config: CrossEncoderConfig | None = None,
    ) -> None:
        """Initialize the InferenceNodeConfig.

        Args:
            node_type: Type of the node.
            module_name: Name of module which is specified as :py:attr:`autointent.modules.base.BaseModule.name`.
            module_config: Hyperparameters of underlying module.
            load_path: Path to the module dump.
            embedder_config: One can override presaved embedder config while loading from file system.
            cross_encoder_config: One can override presaved cross encoder config while loading from file system.
        """
        self.node_type = node_type
        self.module_name = module_name
        self.module_config = module_config
        self.load_path = load_path

        if embedder_config is not None:
            self.embedder_config = embedder_config
        if cross_encoder_config is not None:
            self.cross_encoder_config = cross_encoder_config

    def asdict(self) -> dict[str, Any]:
        """Convert the InferenceNodeConfig to a dictionary.

        Returns:
            A dictionary representation of the InferenceNodeConfig.
        """
        result = {
            "node_type": self.node_type,
            "module_name": self.module_name,
            "module_config": self.module_config,
            "load_path": self.load_path,
        }

        if hasattr(self, "embedder_config"):
            result["embedder_config"] = self.embedder_config.model_dump()
        if hasattr(self, "cross_encoder_config"):
            result["cross_encoder_config"] = self.cross_encoder_config.model_dump()

        return result
