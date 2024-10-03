from dataclasses import dataclass


@dataclass
class LinearScorerConfig:
    _target_: str = "modules.scoring.LinearScorer"
    multilabel: bool
