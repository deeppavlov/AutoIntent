from dataclasses import dataclass
from typing import Any


@dataclass
class ThresholdPredictorConfig:
    thresh: Any  # should be `float | list[float]` but union of containers is not supported by hydra :(
    _target_: str = "autointent.modules.prediction.ThresholdPredictor"
