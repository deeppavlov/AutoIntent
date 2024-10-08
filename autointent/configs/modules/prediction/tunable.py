from dataclasses import dataclass


@dataclass
class TunablePredictorConfig:
    n_trials: int | None = None
    _target_: str = "autointent.modules.prediction.TunablePredictor"
