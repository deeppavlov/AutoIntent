from dataclasses import dataclass


@dataclass
class TunablePredictorConfig:
    n_trials: int
    _target_: str = "autointent.modules.prediction.TunablePredictor"
