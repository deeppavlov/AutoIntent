from dataclasses import dataclass


@dataclass
class ArgmaxPredictorConfig:
    _target_: str = "autointent.modules.prediction.ArgmaxPredictor"
