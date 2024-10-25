from .base import NodeInfo
from .prediction import PredictionNodeInfo
from .regexp import RegExpNodeInfo
from .retrieval import RetrievalNodeInfo
from .scoring import ScoringNodeInfo

NODES_INFO: dict[str, NodeInfo] = {
    "regexp": RegExpNodeInfo(),
    "retrieval": RetrievalNodeInfo(),
    "scoring": ScoringNodeInfo(),
    "prediction": PredictionNodeInfo(),
}

__all__ = ["NodeInfo", "PredictionNodeInfo", "RetrievalNodeInfo", "ScoringNodeInfo", "NODES_INFO"]
