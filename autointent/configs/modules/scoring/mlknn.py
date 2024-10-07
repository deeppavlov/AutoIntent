from dataclasses import dataclass

from autointent.configs.modules.base import ModuleConfig


@dataclass
class MLKnnScorerConfig(ModuleConfig):
    _target_: str = "modules.scoring.MLKnnScorer"
    k: int
    s: float = 1.0
    ignore_first_neighbours: int = 0
