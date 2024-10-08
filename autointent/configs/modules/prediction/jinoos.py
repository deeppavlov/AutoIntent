from dataclasses import dataclass

from autointent.configs.modules.base import ModuleConfig


@dataclass
class JinoosPredictorConfig(ModuleConfig):
    search_space: list[float] | None = None
    _target_: str = "autointent.modules.prediction.JinoosPredictor"
