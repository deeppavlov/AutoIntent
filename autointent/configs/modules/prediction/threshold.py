from dataclasses import dataclass


@dataclass
class ThresholdPredictorConfig:
    thresh: float
    _target_: str = "autointent.modules.prediction.ThresholdPredictor"
