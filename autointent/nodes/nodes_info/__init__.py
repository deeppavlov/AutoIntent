from .base import NodeInfo
from .prediction import PredictionNodeInfo
from .retrieval import RetrievalNodeInfo
from .scoring import ScoringNodeInfo

NODES_INFO: dict[str, NodeInfo] = {
    "retrieval": RetrievalNodeInfo(),
    "scoring": ScoringNodeInfo(),
    "prediction": PredictionNodeInfo(),
}

__all__ = ["NodeInfo", "PredictionNodeInfo", "RetrievalNodeInfo", "ScoringNodeInfo", "NODES_INFO"]
