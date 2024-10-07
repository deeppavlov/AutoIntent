from collections.abc import Callable
from typing import ClassVar

from autointent.metrics import (
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
from autointent.modules import DNNCScorer, KNNScorer, LinearScorer, MLKnnScorer, ScoringModule

from .base import Node


class ScoringNode(Node):
    # todo change type
    metrics_available: ClassVar[dict[str, Callable]] = {
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

    modules_available: ClassVar[dict[str, type[ScoringModule]]] = {
        "knn": KNNScorer,
        "linear": LinearScorer,
        "dnnc": DNNCScorer,
        "mlknn": MLKnnScorer,
    }

    node_type = "scoring"
