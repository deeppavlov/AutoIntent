"""Scoring metrics for multiclass and multilabel classification tasks."""

import logging
from functools import wraps
from typing import Protocol

import numpy as np
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss, roc_auc_score

from ._converter import transform
from .custom_types import LABELS_VALUE_TYPE, SCORES_VALUE_TYPE
from .decision import DecisionMetricFn, decision_accuracy, decision_f1, decision_precision, decision_recall

logger = logging.getLogger(__name__)


class ScoringMetricFn(Protocol):
    """Protocol for scoring metrics.

    Args:
        labels: Ground truth labels for each utterance.
            - multiclass case: list representing an array of shape (n_samples,) with integer values
            - multilabel case: list representing a matrix of shape (n_samples, n_classes) with integer values
        scores: For each utterance, this list contains scores for each of n_classes classes.

    Returns:
        Score of the scoring metric.
    """

    def __call__(self, labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
        """Calculate scoring metric.

        Args:
            labels: Ground truth labels for each utterance.
            scores: Scores for each utterance.

        Returns:
            Score of the scoring metric.
        """
        ...


def ignore_oos(func: ScoringMetricFn) -> ScoringMetricFn:
    """Ignore OOS in metrics calculation (decorator)."""

    @wraps(func)
    def wrapper(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
        labels_filtered = [lab for lab in labels if lab is not None]
        scores_filtered = [score for score, lab in zip(scores, labels, strict=True) if lab is not None]
        return func(labels_filtered, scores_filtered)  # type: ignore[arg-type]

    return wrapper


@ignore_oos
def scoring_log_likelihood(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE, eps: float = 1e-10) -> float:
    r"""Calculate log likelihood score for multiclass and multilabel cases.

    Multiclass case:
    Mean negative cross-entropy for each utterance classification result:

    .. math::

        \frac{1}{\ell}\sum_{i=1}^{\ell}\log(s[y[i]])

    where ``s[y[i]]`` is the predicted score of the ``i``-th utterance having the ground truth label.

    Multilabel case:
    Mean negative binary cross-entropy:

    .. math::

        \frac{1}{\ell}\sum_{i=1}^\ell\sum_{c=1}^C\Big[y[i,c]\cdot\log(s[i,c])+(1-y[i,c])\cdot\log(1-s[i,c])\Big]

    Args:
        labels: Ground truth labels for each utterance.
        scores: For each utterance, a list containing scores for each of n_classes classes.
        eps: A small value to avoid division by zero.

    Returns:
        Score of the scoring metric.

    Raises:
        ValueError: If any scores are not in the range (0,1].
    """
    labels_array, scores_array = transform(labels, scores)
    scores_array[scores_array == 0] = eps

    if np.any((scores_array <= 0) | (scores_array > 1)):
        msg = "One or more scores are not from (0,1]. It is incompatible with `scoring_log_likelihood` metric"
        logger.error(msg)
        raise ValueError(msg)

    if labels_array.ndim == 1:
        relevant_scores = scores_array[np.arange(len(labels_array)), labels_array]
        res = np.mean(np.log(relevant_scores).clip(min=-100, max=100))
    else:
        log_likelihood = labels_array * np.log(scores_array) + (1 - labels_array) * np.log(1 - scores_array)
        clipped_one = log_likelihood.clip(min=-100, max=100)
        res = clipped_one.mean()
    return round(float(res), 6)


@ignore_oos
def scoring_roc_auc(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    r"""Calculate ROC AUC score for multiclass and multilabel cases.

    Macro averaged roc-auc for utterance classification task:

    .. math::

        \frac{1}{C}\sum_{k=1}^C ROCAUC(scores[:, k], labels[:, k])

    where ``C`` is the number of classes.

    Args:
        labels: Ground truth labels for each utterance.
        scores: For each utterance, scores for each of n_classes classes.

    Returns:
        ROC AUC score.
    """
    labels_, scores_ = transform(labels, scores)

    n_classes = scores_.shape[1]
    if labels_.ndim == 1:
        labels_ = (labels_[:, None] == np.arange(n_classes)[None, :]).astype(int)

    return float(roc_auc_score(labels_, scores_, average="macro"))


def _calculate_decision_metric(
    func: DecisionMetricFn, labels: list[int] | list[list[int]], scores: SCORES_VALUE_TYPE
) -> float:
    """Calculate decision metric.

    This function applies the given decision metric function to evaluate the decisions.
    For multiclass classification, decisions are generated using np.argmax.
    For multilabel classification, decisions are generated using a threshold of 0.5.

    Args:
        func: Decision metric function.
        labels: Ground truth labels for each utterance.
        scores: For each utterance, scores for each of n_classes classes.

    Returns:
        Score of the decision metric.
    """
    if isinstance(labels[0], int):
        pred_labels = np.argmax(scores, axis=1).tolist()
        res = func(labels, pred_labels)
    else:
        pred_labels = (np.array(scores) > 0.5).astype(int).tolist()  # noqa: PLR2004
        res = func(labels, pred_labels)

    return res


@ignore_oos
def scoring_accuracy(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """Calculate accuracy for multiclass and multilabel classification.

    Uses decision_accuracy to evaluate decisions.

    Args:
        labels: Ground truth labels for each utterance.
        scores: For each utterance, scores for each of n_classes classes.

    Returns:
        Classification accuracy score.
    """
    return _calculate_decision_metric(decision_accuracy, labels, scores)


@ignore_oos
def scoring_f1(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """Calculate F1 score for multiclass and multilabel classification.

    Uses decision_f1 to evaluate decisions.

    Args:
        labels: Ground truth labels for each sample.
        scores: For each sample, scores for each of n_classes classes.

    Returns:
        F1 score.
    """
    return _calculate_decision_metric(decision_f1, labels, scores)


@ignore_oos
def scoring_precision(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """Calculate precision for multiclass and multilabel classification.

    Uses decision_precision to evaluate decisions.

    Args:
        labels: Ground truth labels for each sample.
        scores: For each sample, scores for each of n_classes classes.

    Returns:
        Precision score.
    """
    return _calculate_decision_metric(decision_precision, labels, scores)


@ignore_oos
def scoring_recall(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """Calculate recall for multiclass and multilabel classification.

    Uses decision_recall to evaluate decisions.

    Args:
        labels: Ground truth labels for each sample.
        scores: For each sample, scores for each of n_classes classes.

    Returns:
        Recall score.
    """
    return _calculate_decision_metric(decision_recall, labels, scores)


@ignore_oos
def scoring_hit_rate(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    r"""Calculate hit rate for multilabel classification.

    Hit rate measures the fraction of cases where the top-ranked label is in the set
    of true labels for the instance.

    .. math::

        \text{Hit Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}(y_{\text{top},i} \in y_{\text{true},i})

    Args:
        labels: Ground truth labels for each sample.
        scores: For each sample, scores for each of n_classes classes.

    Returns:
        Hit rate score.
    """
    labels_, scores_ = transform(labels, scores)

    top_ranked_labels = np.argmax(scores_, axis=1)
    is_in = labels_[np.arange(len(labels)), top_ranked_labels]

    return float(np.mean(is_in))


@ignore_oos
def scoring_neg_coverage(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """Calculate negative coverage for multilabel classification.

    Evaluates how far we need to go down the list of labels to cover all proper labels.
    The ideal value is 1, the worst value is 0.

    Args:
        labels: Ground truth labels for each utterance.
        scores: For each utterance, scores for each of n_classes classes.

    Returns:
        Negative coverage score.
    """
    labels_, scores_ = transform(labels, scores)

    n_classes = scores_.shape[1]
    return float(1 - (coverage_error(labels, scores) - 1) / (n_classes - 1))


@ignore_oos
def scoring_neg_ranking_loss(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """Calculate negative ranking loss for multilabel classification.

    Computes average number of incorrectly ordered label pairs weighted by label set size.
    The ideal value is 0.

    Args:
        labels: Ground truth labels for each utterance.
        scores: For each utterance, scores for each of n_classes classes.

    Returns:
        Negative ranking loss score.
    """
    return float(-label_ranking_loss(labels, scores))


@ignore_oos
def scoring_map(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """Calculate mean average precision (MAP) score for multilabel classification.

    Measures precision at different ranking levels, averaged across all queries.
    The ideal value is 1, indicating perfect ranking.

    Args:
        labels: Ground truth labels for each sample.
        scores: For each sample, scores for each of n_classes classes.

    Returns:
        Mean average precision score.
    """
    return float(label_ranking_average_precision_score(labels, scores))
