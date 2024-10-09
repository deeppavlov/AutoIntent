from dataclasses import dataclass

from autointent.configs.modules.base import ModuleConfig


@dataclass
class TunablePredictorConfig(ModuleConfig):
    n_trials: int | None = None
    _target_: str = "autointent.modules.prediction.TunablePredictor"
