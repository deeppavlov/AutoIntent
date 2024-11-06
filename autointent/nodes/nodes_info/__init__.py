from .base import NodeInfo
from .prediction import PredictionNodeInfo
from .regexp import RegExpNodeInfo
from .retrieval import RetrievalNodeInfo
from .scoring import ScoringNodeInfo

NODES_INFO: dict[str, NodeInfo] = {
    "retrieval": RetrievalNodeInfo(),
    "scoring": ScoringNodeInfo(),
    "prediction": PredictionNodeInfo(),
    "regexp": RegExpNodeInfo(),
}

__all__ = [
    "NodeInfo",
    "PredictionNodeInfo",
    "RegExpNodeInfo",
    "RetrievalNodeInfo",
    "ScoringNodeInfo",
    "NODES_INFO",
]
