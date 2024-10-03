from dataclasses import dataclass


@dataclass
class MLKnnScorerConfig:
    _target_: str = "modules.scoring.MLKnnScorer"
    k: int
    s: float = 1.0
    ignore_first_neighbours: int = 0
