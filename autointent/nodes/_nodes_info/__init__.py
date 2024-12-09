from autointent.custom_types import NodeType

from ._base import NodeInfo
from ._decision import DecisionNodeInfo
from ._regexp import RegExpNodeInfo
from ._retrieval import RetrievalNodeInfo
from ._scoring import ScoringNodeInfo

NODES_INFO: dict[str, NodeInfo] = {
    NodeType.retrieval: RetrievalNodeInfo(),
    NodeType.scoring: ScoringNodeInfo(),
    NodeType.decision: DecisionNodeInfo(),
    NodeType.regexp: RegExpNodeInfo(),
}

__all__ = [
    "NODES_INFO",
    "DecisionNodeInfo",
    "NodeInfo",
    "RegExpNodeInfo",
    "RetrievalNodeInfo",
    "ScoringNodeInfo",
]
