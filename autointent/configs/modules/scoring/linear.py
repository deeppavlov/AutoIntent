from dataclasses import dataclass


@dataclass
class LinearScorerConfig:
    _target_: str = "autointent.modules.scoring.LinearScorer"
