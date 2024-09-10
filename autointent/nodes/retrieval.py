from ..metrics import (
    retrieval_hit_rate,
    retrieval_hit_rate_macro,
    retrieval_hit_rate_tolerant,
    retrieval_map,
    retrieval_map_macro,
    retrieval_map_tolerant,
    retrieval_mrr,
    retrieval_mrr_macro,
    retrieval_mrr_tolerant,
    retrieval_ndcg,
    retrieval_ndcg_macro,
    retrieval_ndcg_tolerant,
    retrieval_precision,
    retrieval_precision_macro,
    retrieval_precision_tolerant,
)
from ..modules import VectorDBModule
from .base import Node


class RetrievalNode(Node):
    metrics_available = {
        "retrieval_hit_rate": retrieval_hit_rate,
        "retrieval_hit_rate_macro": retrieval_hit_rate_macro,
        "retrieval_hit_rate_tolerant": retrieval_hit_rate_tolerant,
        "retrieval_map": retrieval_map,
        "retrieval_map_macro": retrieval_map_macro,
        "retrieval_map_tolerant": retrieval_map_tolerant,
        "retrieval_mrr": retrieval_mrr,
        "retrieval_mrr_macro": retrieval_mrr_macro,
        "retrieval_mrr_tolerant": retrieval_mrr_tolerant,
        "retrieval_ndcg": retrieval_ndcg,
        "retrieval_ndcg_macro": retrieval_ndcg_macro,
        "retrieval_ndcg_tolerant": retrieval_ndcg_tolerant,
        "retrieval_precision": retrieval_precision,
        "retrieval_precision_macro": retrieval_precision_macro,
        "retrieval_precision_tolerant": retrieval_precision_tolerant,
    }

    modules_available = {"vector_db": VectorDBModule}

    node_type = "retrieval"
