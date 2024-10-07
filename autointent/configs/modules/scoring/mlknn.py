from dataclasses import dataclass


@dataclass
class MLKnnScorerConfig:
    k: int
    s: float = 1.0
    ignore_first_neighbours: int = 0
    _target_: str = "autointent.modules.scoring.MLKnnScorer"
