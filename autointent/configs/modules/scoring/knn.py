from dataclasses import dataclass
from enum import Enum

from autointent.configs.modules.base import ModuleConfig


class KNNWeightsType(Enum):
    UNIFORM = "uniform"
    DISTANCE = "distance"
    CLOSEST = "closest"


@dataclass
class KNNScorerConfig(ModuleConfig):
    _target_: str = "modules.scoring.KNNScorer"
    k: int
    weights: KNNWeightsType
