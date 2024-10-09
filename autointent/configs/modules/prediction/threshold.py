from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING

from autointent.configs.modules.base import ModuleConfig


@dataclass
class ThresholdPredictorConfig(ModuleConfig):
    thresh: Any = MISSING  # should be `float | list[float]` but union of containers is not supported by hydra :(
    _target_: str = "autointent.modules.prediction.ThresholdPredictor"
