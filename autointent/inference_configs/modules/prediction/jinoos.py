from dataclasses import dataclass


@dataclass
class JinoosPredictorConfig:
    _target_: str = "modules.prediction.JinoosPredictor"
    search_space: list[float] | None
