from dataclasses import dataclass

from omegaconf import MISSING

from autointent.configs.modules.base import ModuleConfig


@dataclass
class KNNScorerConfig(ModuleConfig):
    k: int = MISSING
    weights: str = MISSING
    _target_: str = "autointent.modules.scoring.KNNScorer"
