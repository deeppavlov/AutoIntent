from dataclasses import dataclass
from enum import Enum


class KNNWeightsType(Enum):
    UNIFORM = "uniform"
    DISTANCE = "distance"
    CLOSEST = "closest"


@dataclass
class KNNScorerConfig:
    _target_: str = "modules.scoring.KNNScorer"
    k: int
    weights: KNNWeightsType
