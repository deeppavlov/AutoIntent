from autointent.custom_types import NodeType

from .base import NodeInfo
from .prediction import PredictionNodeInfo
from .retrieval import RetrievalNodeInfo
from .scoring import ScoringNodeInfo

NODES_INFO: dict[str, NodeInfo] = {
    NodeType.retrieval: RetrievalNodeInfo(),
    NodeType.scoring: ScoringNodeInfo(),
    NodeType.prediction: PredictionNodeInfo(),
}

__all__ = ["NodeInfo", "PredictionNodeInfo", "RetrievalNodeInfo", "ScoringNodeInfo", "NODES_INFO"]
