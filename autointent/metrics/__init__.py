"""All the metrics for regex, retrieval, scoring and decision nodes."""

from autointent._utils import _funcs_to_dict

from .decision import (
    DecisionMetricFn,
    decision_accuracy,
    decision_f1,
    decision_precision,
    decision_recall,
    decision_roc_auc,
)
from .regex import RegexMetricFn, regex_partial_accuracy, regex_partial_precision
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

SCORING_METRICS_MULTILABEL: dict[str, ScoringMetricFn] = _funcs_to_dict(
    # multiclass except for scoring_roc_auc
    scoring_accuracy,
    scoring_f1,
    scoring_log_likelihood,
    scoring_precision,
    scoring_recall,
    # multilabel
    scoring_hit_rate,
    scoring_map,
    scoring_neg_coverage,
    scoring_neg_ranking_loss,
)

DECISION_METRICS: dict[str, DecisionMetricFn] = _funcs_to_dict(
    decision_accuracy,
    decision_f1,
    decision_precision,
    decision_recall,
    decision_roc_auc,
)

DICISION_METRICS_MULTILABEL: dict[str, DecisionMetricFn] = _funcs_to_dict(
    decision_accuracy,
    decision_f1,
    decision_precision,
    decision_recall,
)

REGEX_METRICS = _funcs_to_dict(regex_partial_accuracy, regex_partial_precision)

METRIC_FN = DecisionMetricFn | RegexMetricFn | RetrievalMetricFn | ScoringMetricFn

__all__ = [
    "METRIC_FN",
    "DecisionMetricFn",
    "RegexMetricFn",
    "RetrievalMetricFn",
    "ScoringMetricFn",
    "decision_accuracy",
    "decision_f1",
    "decision_precision",
    "decision_recall",
    "decision_roc_auc",
    "regex_partial_accuracy",
    "regex_partial_precision",
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
