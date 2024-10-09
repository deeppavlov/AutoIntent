from .prediction import (
    PredictionMetricFn,
    prediction_accuracy,
    prediction_f1,
    prediction_precision,
    prediction_recall,
    prediction_roc_auc,
)
from .regexp import regexp_partial_accuracy, regexp_partial_precision
from .retrieval import (
    RetrievalMetricFn,
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
from .scoring import (
    ScoringMetricFn,
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

RETRIEVAL_METRICS_MULTICLASS: dict[str, RetrievalMetricFn] = {
    "retrieval_hit_rate": retrieval_hit_rate,
    "retrieval_map": retrieval_map,
    "retrieval_mrr": retrieval_mrr,
    "retrieval_ndcg": retrieval_ndcg,
    "retrieval_precision": retrieval_precision,
}

RETRIEVAL_METRICS_MULTILABEL: dict[str, RetrievalMetricFn] = {
    "retrieval_hit_rate_intersecting": retrieval_hit_rate_intersecting,
    "retrieval_hit_rate_macro": retrieval_hit_rate_macro,
    "retrieval_map_intersecting": retrieval_map_intersecting,
    "retrieval_map_macro": retrieval_map_macro,
    "retrieval_mrr_intersecting": retrieval_mrr_intersecting,
    "retrieval_mrr_macro": retrieval_mrr_macro,
    "retrieval_ndcg_intersecting": retrieval_ndcg_intersecting,
    "retrieval_ndcg_macro": retrieval_ndcg_macro,
    "retrieval_precision_intersecting": retrieval_precision_intersecting,
    "retrieval_precision_macro": retrieval_precision_macro,
}

SCORING_METRICS_MULTICLASS: dict[str, ScoringMetricFn] = {
    "scoring_accuracy": scoring_accuracy,
    "scoring_f1": scoring_f1,
    "scoring_log_likelihood": scoring_log_likelihood,
    "scoring_precision": scoring_precision,
    "scoring_recall": scoring_recall,
    "scoring_roc_auc": scoring_roc_auc,
}

SCORING_METRICS_MULTILABEL: dict[str, ScoringMetricFn] = SCORING_METRICS_MULTICLASS | {
    "scoring_hit_rate": scoring_hit_rate,
    "scoring_map": scoring_map,
    "scoring_neg_coverage": scoring_neg_coverage,
    "scoring_neg_ranking_loss": scoring_neg_ranking_loss,
}

PREDICTION_METRICS_MULTICLASS: dict[str, PredictionMetricFn] = {
    "prediction_accuracy": prediction_accuracy,
    "prediction_f1": prediction_f1,
    "prediction_precision": prediction_precision,
    "prediction_recall": prediction_recall,
    "prediction_roc_auc": prediction_roc_auc,
}
PREDICTION_METRICS_MULTILABEL = PREDICTION_METRICS_MULTICLASS
__all__ = [
    "PredictionMetricFn",
    "prediction_accuracy",
    "prediction_f1",
    "prediction_precision",
    "prediction_recall",
    "prediction_roc_auc",
    "regexp_partial_accuracy",
    "regexp_partial_precision",
    "RetrievalMetricFn",
    "retrieval_hit_rate",
    "retrieval_hit_rate_intersecting",
    "retrieval_hit_rate_macro",
    "retrieval_map",
    "retrieval_map_intersecting",
    "retrieval_map_macro",
    "retrieval_mrr",
    "retrieval_mrr_intersecting",
    "retrieval_mrr_macro",
    "retrieval_ndcg",
    "retrieval_ndcg_intersecting",
    "retrieval_ndcg_macro",
    "retrieval_precision",
    "retrieval_precision_intersecting",
    "retrieval_precision_macro",
    "ScoringMetricFn",
    "scoring_accuracy",
    "scoring_f1",
    "scoring_hit_rate",
    "scoring_log_likelihood",
    "scoring_map",
    "scoring_neg_coverage",
    "scoring_neg_ranking_loss",
    "scoring_precision",
    "scoring_recall",
    "scoring_roc_auc",
]
