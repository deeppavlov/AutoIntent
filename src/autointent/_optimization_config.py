from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from .configs import (
    CrossEncoderConfig,
    DataConfig,
    HFModelConfig,
    HPOConfig,
    LoggingConfig,
    initialize_embedder_config,
)

if TYPE_CHECKING:
    from pydantic import PositiveInt

    from .configs import (
        EmbedderConfig,
    )


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

    transformer_config: HFModelConfig = HFModelConfig()

    hpo_config: HPOConfig = HPOConfig()

    seed: PositiveInt = 42
