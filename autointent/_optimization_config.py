from pydantic import BaseModel, PositiveInt

from .configs import DataConfig, LoggingConfig
from .custom_types import SamplerType
from .nodes.schemes import OptimizationSearchSpaceConfig


class OptimizationConfig(BaseModel):
    """Configuration for the optimization process."""

    data_config: DataConfig = DataConfig()
    search_space: OptimizationSearchSpaceConfig
    logging_config: LoggingConfig = LoggingConfig()
    sampler: SamplerType = "brute"
    seed: PositiveInt = 42
