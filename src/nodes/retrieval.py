from ..metrics import (
    retrieval_hit_rate,
    retrieval_map,
    retrieval_mrr,
    retrieval_ndcg,
    retrieval_precision,
)
from ..modules import VectorDBModule
from .base import Node


class RetrievalNode(Node):
    metrics_available = {
        "retrieval_map": retrieval_map,
        "retrieval_ndcg": retrieval_ndcg,
        "retrieval_hit_rate": retrieval_hit_rate,
        "retrieval_precision": retrieval_precision,
        "retrieval_mrr": retrieval_mrr,
    }

    modules_available = {"vector_db": VectorDBModule}
