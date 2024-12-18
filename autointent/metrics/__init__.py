"""All the metrics for regexp, retrieval, scoring and decision nodes."""

from autointent._utils import _funcs_to_dict

from .decision import (
    DecisionMetricFn,
    decision_accuracy,
    decision_f1,
    decision_precision,
    decision_recall,
    decision_roc_auc,
)
from .regexp import RegexpMetricFn, regexp_partial_accuracy, regexp_partial_precision
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

RETRIEVAL_METRICS_MULTICLASS: dict[str, RetrievalMetricFn] = _funcs_to_dict(
    retrieval_hit_rate,
    retrieval_map,
    retrieval_mrr,
    retrieval_ndcg,
    retrieval_precision,
)

RETRIEVAL_METRICS_MULTILABEL: dict[str, RetrievalMetricFn] = _funcs_to_dict(
    retrieval_hit_rate_intersecting,
    retrieval_hit_rate_macro,
    retrieval_map_intersecting,
    retrieval_map_macro,
    retrieval_mrr_intersecting,
    retrieval_mrr_macro,
    retrieval_ndcg_intersecting,
    retrieval_ndcg_macro,
    retrieval_precision_intersecting,
    retrieval_precision_macro,
)

SCORING_METRICS_MULTICLASS: dict[str, ScoringMetricFn] = _funcs_to_dict(
    scoring_accuracy,
    scoring_f1,
    scoring_log_likelihood,
    scoring_precision,
    scoring_recall,
    scoring_roc_auc,
)

SCORING_METRICS_MULTILABEL: dict[str, ScoringMetricFn] = SCORING_METRICS_MULTICLASS | _funcs_to_dict(
    scoring_hit_rate,
    scoring_map,
    scoring_neg_coverage,
    scoring_neg_ranking_loss,
)

PREDICTION_METRICS_MULTICLASS: dict[str, DecisionMetricFn] = _funcs_to_dict(
    decision_accuracy,
    decision_f1,
    decision_precision,
    decision_recall,
    decision_roc_auc,
)

PREDICTION_METRICS_MULTILABEL = PREDICTION_METRICS_MULTICLASS

REGEXP_METRICS = _funcs_to_dict(regexp_partial_accuracy, regexp_partial_precision)

METRIC_FN = DecisionMetricFn | RegexpMetricFn | RetrievalMetricFn | ScoringMetricFn

__all__ = [
    "METRIC_FN",
    "DecisionMetricFn",
    "RegexpMetricFn",
    "RetrievalMetricFn",
    "ScoringMetricFn",
    "decision_accuracy",
    "decision_f1",
    "decision_precision",
    "decision_recall",
    "decision_roc_auc",
    "regexp_partial_accuracy",
    "regexp_partial_precision",
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
