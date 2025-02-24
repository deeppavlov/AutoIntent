from pydantic import BaseModel, PositiveInt

from .configs import CrossEncoderConfig, DataConfig, EmbedderConfig, LoggingConfig
from .custom_types import SamplerType
from .nodes.schemes import OptimizationSearchSpaceConfig


class OptimizationConfig(BaseModel):
    """Configuration for the optimization process."""

    data_config: DataConfig = DataConfig()
    search_space: OptimizationSearchSpaceConfig
    logging_config: LoggingConfig = LoggingConfig()
    embedder_config: EmbedderConfig = EmbedderConfig()
    cross_encoder_config: CrossEncoderConfig = CrossEncoderConfig()
    sampler: SamplerType = "brute"
    seed: PositiveInt = 42
