from autointent.custom_types import NodeType

from ._base import NodeInfo
from ._decision import DecisionNodeInfo
from ._embedding import EmbeddingNodeInfo
from ._regex import RegexNodeInfo
from ._scoring import ScoringNodeInfo

NODES_INFO: dict[str, NodeInfo] = {
    NodeType.embedding: EmbeddingNodeInfo(),
    NodeType.scoring: ScoringNodeInfo(),
    NodeType.decision: DecisionNodeInfo(),
    NodeType.regex: RegexNodeInfo(),
}

__all__ = ["NODES_INFO", "DecisionNodeInfo", "EmbeddingNodeInfo", "NodeInfo", "RegexNodeInfo", "ScoringNodeInfo"]
