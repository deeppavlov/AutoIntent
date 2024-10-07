from dataclasses import dataclass


@dataclass
class JinoosPredictorConfig:
    search_space: list[float] | None = None
    _target_: str = "autointent.modules.prediction.JinoosPredictor"
