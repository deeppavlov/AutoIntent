from dataclasses import dataclass

# from enum import Enum
# class KNNWeightsType(Enum):
#     UNIFORM = "uniform"
#     DISTANCE = "distance"
#     CLOSEST = "closest"


@dataclass
class KNNScorerConfig:
    k: int
    weights: str
    _target_: str = "autointent.modules.scoring.KNNScorer"
