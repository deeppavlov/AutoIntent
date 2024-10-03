from dataclasses import dataclass


@dataclass
class ThresholdPredictorConfig:
    _target_: str = "modules.prediction.ThresholdPredictor"
    thresh: float | list[float]
