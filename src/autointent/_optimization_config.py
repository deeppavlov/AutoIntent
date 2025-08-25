from typing import Any

from pydantic import BaseModel, PositiveInt

from .configs import CrossEncoderConfig, DataConfig, EmbedderConfig, HFModelConfig, HPOConfig, LoggingConfig


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

    embedder_config: EmbedderConfig = EmbedderConfig()

    cross_encoder_config: CrossEncoderConfig = CrossEncoderConfig()

    transformer_config: HFModelConfig = HFModelConfig()

    hpo_config: HPOConfig = HPOConfig()

    seed: PositiveInt = 42
