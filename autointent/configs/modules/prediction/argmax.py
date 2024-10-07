from dataclasses import dataclass

from autointent.configs.modules.base import ModuleConfig


@dataclass
class ArgmaxPredictorConfig(ModuleConfig):
    _target_: str = "modules.prediction.ArgmaxPredictor"
