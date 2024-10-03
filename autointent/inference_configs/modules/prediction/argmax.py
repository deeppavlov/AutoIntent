from dataclasses import dataclass


@dataclass
class ArgmaxPredictorConfig:
    _target_: str = "modules.prediction.ArgmaxPredictor"
