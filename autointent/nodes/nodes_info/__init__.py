from autointent.custom_types import NodeType

from .base import NodeInfo
from .prediction import PredictionNodeInfo
from .regexp import RegExpNodeInfo
from .retrieval import RetrievalNodeInfo
from .scoring import ScoringNodeInfo

NODES_INFO: dict[str, NodeInfo] = {
    NodeType.retrieval: RetrievalNodeInfo(),
    NodeType.scoring: ScoringNodeInfo(),
    NodeType.prediction: PredictionNodeInfo(),
    NodeType.regexp: RegExpNodeInfo(),
}

__all__ = [
    "NodeInfo",
    "PredictionNodeInfo",
    "RegExpNodeInfo",
    "RetrievalNodeInfo",
    "ScoringNodeInfo",
    "NODES_INFO",
]
