"""Configuration for the nodes."""

from dataclasses import asdict, dataclass
from typing import Any

from autointent.custom_types import NodeType

from ._transformers import CrossEncoderConfig, EmbedderConfig


@dataclass
class InferenceNodeConfig:
    """Configuration for the inference node."""

    node_type: NodeType
    """Type of the node."""
    module_name: str
    """Name of module which is specified as :py:attr:`autointent.modules.base.BaseModule.name`."""
    module_config: dict[str, Any]
    """Hyperparameters of underlying module."""
    load_path: str
    """Path to the module dump."""
    embedder_config: EmbedderConfig | None = None
    """One can override presaved embedder config while loading from file system."""
    cross_encoder_config: CrossEncoderConfig | None = None
    """One can override presaved cross encoder config while loading from file system."""

    def asdict(self) -> dict[str, Any]:
        """Convert config to dict format."""
        res = asdict(self)
        if self.embedder_config is not None:
            res["embedder_config"] = self.embedder_config.model_dump()
        else:
            res.pop("embedder_config")
        if self.cross_encoder_config is not None:
            res["cross_encoder_config"] = self.cross_encoder_config.model_dump()
        else:
            res.pop("cross_encoder_config")
        return res
