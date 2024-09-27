from autointent.metrics import (
    retrieval_hit_rate,
    retrieval_hit_rate_intersecting,
    retrieval_hit_rate_macro,
    retrieval_map,
    retrieval_map_intersecting,
    retrieval_map_macro,
    retrieval_mrr,
    retrieval_mrr_intersecting,
    retrieval_mrr_macro,
    retrieval_ndcg,
    retrieval_ndcg_intersecting,
    retrieval_ndcg_macro,
    retrieval_precision,
    retrieval_precision_intersecting,
    retrieval_precision_macro,
)
from autointent.modules import VectorDBModule

from .base import Node


class RetrievalNode(Node):
    metrics_available = {
        "retrieval_hit_rate": retrieval_hit_rate,
        "retrieval_hit_rate_macro": retrieval_hit_rate_macro,
        "retrieval_hit_rate_intersecting": retrieval_hit_rate_intersecting,
        "retrieval_map": retrieval_map,
        "retrieval_map_macro": retrieval_map_macro,
        "retrieval_map_intersecting": retrieval_map_intersecting,
        "retrieval_mrr": retrieval_mrr,
        "retrieval_mrr_macro": retrieval_mrr_macro,
        "retrieval_mrr_intersecting": retrieval_mrr_intersecting,
        "retrieval_ndcg": retrieval_ndcg,
        "retrieval_ndcg_macro": retrieval_ndcg_macro,
        "retrieval_ndcg_intersecting": retrieval_ndcg_intersecting,
        "retrieval_precision": retrieval_precision,
        "retrieval_precision_macro": retrieval_precision_macro,
        "retrieval_precision_intersecting": retrieval_precision_intersecting,
    }

    modules_available = {"vector_db": VectorDBModule}

    node_type = "retrieval"
