from dataclasses import dataclass

from autointent.configs.modules.base import ModuleConfig


@dataclass
class ThresholdPredictorConfig(ModuleConfig):
    _target_: str = "modules.prediction.ThresholdPredictor"
    thresh: float | list[float]
