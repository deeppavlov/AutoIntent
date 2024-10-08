from dataclasses import dataclass


@dataclass
class ThresholdPredictorConfig:
    thresh: float | list[float]  # not supported by hydra :(
    _target_: str = "autointent.modules.prediction.ThresholdPredictor"
