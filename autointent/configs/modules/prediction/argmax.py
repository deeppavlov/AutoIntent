from dataclasses import dataclass

from autointent.configs.modules.base import ModuleConfig


@dataclass
class ArgmaxPredictorConfig(ModuleConfig):
    _target_: str = "autointent.modules.prediction.ArgmaxPredictor"
