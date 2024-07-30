from ..metrics import scoring_neg_cross_entropy, scoring_roc_auc
from ..modules import DNNCScorer, KNNScorer, LinearScorer
from .base import Node


class ScoringNode(Node):
    metrics_available = {
        "neg_cross_entropy": scoring_neg_cross_entropy,
        "roc_auc": scoring_roc_auc,
    }

    modules_available = {"knn": KNNScorer, "linear": LinearScorer, "dnnc": DNNCScorer}
