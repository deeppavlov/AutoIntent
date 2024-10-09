from dataclasses import dataclass

from omegaconf import MISSING

from autointent.configs.modules.base import ModuleConfig


@dataclass
class MLKnnScorerConfig(ModuleConfig):
    k: int = MISSING
    s: float = 1.0
    ignore_first_neighbours: int = 0
    _target_: str = "autointent.modules.scoring.MLKnnScorer"
