from ..metrics import (
    scoring_accuracy,
    scoring_f1,
    scoring_hit_rate,
    scoring_log_likelihood,
    scoring_map,
    scoring_neg_coverage,
    scoring_neg_ranking_loss,
    scoring_precision,
    scoring_recall,
    scoring_roc_auc,
)
from ..modules import DNNCScorer, KNNScorer, LinearScorer
from .base import Node


class ScoringNode(Node):
    metrics_available = {
        "scoring_log_likelihood": scoring_log_likelihood,
        "scoring_roc_auc": scoring_roc_auc,
        "scoring_accuracy": scoring_accuracy,
        "scoring_f1": scoring_f1,
        "scoring_precision": scoring_precision,
        "scoring_recall": scoring_recall,
        "scoring_neg_ranking_loss": scoring_neg_ranking_loss,
        "scoring_neg_coverage": scoring_neg_coverage,
        "scoring_hit_rate": scoring_hit_rate,
        "scoring_map": scoring_map,
    }

    modules_available = {"knn": KNNScorer, "linear": LinearScorer, "dnnc": DNNCScorer}

    node_type = "scoring"
