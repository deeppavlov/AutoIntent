from dataclasses import dataclass


@dataclass
class TunablePredictorConfig:
    _target_: str = "modules.prediction.TunablePredictor"
