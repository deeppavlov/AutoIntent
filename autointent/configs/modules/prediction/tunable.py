from dataclasses import dataclass

from autointent.configs.modules.base import ModuleConfig


@dataclass
class TunablePredictorConfig(ModuleConfig):
    _target_: str = "modules.prediction.TunablePredictor"
