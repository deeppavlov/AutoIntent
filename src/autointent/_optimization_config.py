from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PositiveInt, field_validator

from .configs import (
    CrossEncoderConfig,
    DataConfig,
    EmbedderConfig,
    HFModelConfig,
    HPOConfig,
    LoggingConfig,
    get_default_hfmodel_config,
    initialize_embedder_config,
)
from .utils import load_preset

if TYPE_CHECKING:
    from .custom_types import SearchSpacePreset


class OptimizationConfig(BaseModel):
    """Configuration for the optimization process.

    One can use it to customize optimization beyond choosing different preset.
    Instantiate it and pass to :py:meth:`autointent.Pipeline.from_optimization_config`.
    """

    data_config: DataConfig = DataConfig()

    search_space: list[dict[str, Any]]
    """See tutorial on search space customization."""

    logging_config: LoggingConfig = LoggingConfig()
    """See tutorial on logging configuration."""

    embedder_config: EmbedderConfig = Field(default_factory=lambda: initialize_embedder_config(None))

    @field_validator("embedder_config", mode="before")
    @classmethod
    def validate_embedder_config(cls, v: Any) -> EmbedderConfig:  # noqa: ANN401
        """Validate and convert embedder config to proper type."""
        return initialize_embedder_config(v)

    cross_encoder_config: CrossEncoderConfig = CrossEncoderConfig()

    transformer_config: HFModelConfig = get_default_hfmodel_config()

    hpo_config: HPOConfig = HPOConfig()

    seed: PositiveInt = 42

    @classmethod
    def from_preset(cls, preset: SearchSpacePreset) -> OptimizationConfig:
        return cls.model_validate(load_preset(preset))
