from dataclasses import dataclass

from autointent.configs.modules.base import ModuleConfig


@dataclass
class JinoosPredictorConfig(ModuleConfig):
    _target_: str = "modules.prediction.JinoosPredictor"
    search_space: list[float] | None
