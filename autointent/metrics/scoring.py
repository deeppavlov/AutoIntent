"""Scoring metrics for multiclass and multilabel classification tasks."""

import logging
from typing import Protocol

import numpy as np
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss, roc_auc_score

from ._converter import transform
from ._custom_types import LABELS_VALUE_TYPE, SCORES_VALUE_TYPE
from .prediction import PredictionMetricFn, prediction_accuracy, prediction_f1, prediction_precision, prediction_recall

logger = logging.getLogger(__name__)


class ScoringMetricFn(Protocol):
    """Protocol for scoring metrics."""

    def __call__(self, labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
        """
        Calculate scoring metric.

        :param labels: ground truth labels for each utterance
            - multiclass case: list representing an array of shape `(n_samples,)` with integer values
            - multilabel case: list representing a matrix of shape `(n_samples, n_classes)` with integer values
        :param scores: for each utterance, this list contains scores for each of `n_classes` classes
        :return: Score of the scoring metric
        """
        ...


def scoring_log_likelihood(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE, eps: float = 1e-10) -> float:
    r"""
    Supports multiclass and multilabel cases.

    Multiclass case
    Mean negative cross-entropy for each utterance classification result:

    .. math::

        \\frac{1}{\\ell}\\sum_{i=1}^{\\ell}\\log(s[y[i]])

    where ``s[y[i]]`` is a predicted score of ``i``\\ th utterance having ground truth label

    Multilabel case
    Mean negative binary cross-entropy:

    .. math::

        \\frac{1}{\\ell}\\sum_{i=1}^\\ell\\sum_{c=1}^C\\Big[y[i,c]\\cdot\\log(s[i,c])+(1-y[i,c])\\cdot\\log(1-s[i,c])\\Big]

    where ``s[i,c]`` is a predicted score of ``i``\\ th utterance having ground truth label ``c``

    :param labels: ground truth labels for each utterance
    :param scores: for each utterance, this list contains scores for each of `n_classes` classes
    :param eps: small value to avoid division by zero
    :return: Score of the scoring metric
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
    return res  # type: ignore[no-any-return]


def scoring_roc_auc(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    r"""
    Supports multiclass and multilabel cases.

    Macro averaged roc-auc for utterance classification task, i.e.

    .. math::

        \frac{1}{C}\\sum_{k=1}^C ROCAUC(scores[:, k], labels[:, k])

    where ``C`` is the number of classes

    :param labels: ground truth labels for each utterance
    :param scores: for each utterance, this list contains scores for each of `n_classes` classes
    :return: Score of the scoring metric
    """
    labels_, scores_ = transform(labels, scores)

    n_classes = scores_.shape[1]
    if labels_.ndim == 1:
        labels_ = (labels_[:, None] == np.arange(n_classes)[None, :]).astype(int)

    return roc_auc_score(labels_, scores_, average="macro")  # type: ignore[no-any-return]


def _calculate_prediction_metric(
    func: PredictionMetricFn, labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE
) -> float:
    """
    Calculate prediction metric.

    :param func: prediction metric function
    :param labels: ground truth labels for each utterance
    :param scores: for each utterance, this list contains scores for each of `n_classes` classes
    :return: Score of the scoring metric
    """
    labels_, scores_ = transform(labels, scores)

    if labels_.ndim == 1:
        pred_labels = np.argmax(scores, axis=1)
        res = func(labels_, pred_labels)
    else:
        pred_labels = (scores_ > 0.5).astype(int)  # noqa: PLR2004
        res = func(labels_, pred_labels)

    return res


def scoring_accuracy(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """
    supports multiclass and multilabel.

    :param labels: ground truth labels for each utterance
    :param scores: for each utterance, this list contains scores for each of `n_classes` classes
    :return: Score of the scoring metric
    """
    return _calculate_prediction_metric(prediction_accuracy, labels, scores)


def scoring_f1(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """
    supports multiclass and multilabel.

    :param labels: ground truth labels for each utterance
    :param scores: for each utterance, this list contains scores for each of `n_classes` classes
    :return: Score of the scoring metric
    """
    return _calculate_prediction_metric(prediction_f1, labels, scores)


def scoring_precision(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """
    supports multiclass and multilabel.

    :param labels: ground truth labels for each utterance
    :param scores: for each utterance, this list contains scores for each of `n_classes` classes
    :return: Score of the scoring metric
    """
    return _calculate_prediction_metric(prediction_precision, labels, scores)


def scoring_recall(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """
    supports multiclass and multilabel.

    :param labels: ground truth labels for each utterance
    :param scores: for each utterance, this list contains scores for each of `n_classes` classes
    :return: Score of the scoring metric
    """
    return _calculate_prediction_metric(prediction_recall, labels, scores)


def scoring_hit_rate(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """
    supports multilabel.

    calculates fraction of cases when the top-ranked label is in the set of proper labels of the instance

    :param labels: ground truth labels for each utterance
    :param scores: for each utterance, this list contains scores for each of `n_classes` classes
    :return: Score of the scoring metric
    """
    labels_, scores_ = transform(labels, scores)

    top_ranked_labels = np.argmax(scores_, axis=1)
    is_in = labels_[np.arange(len(labels)), top_ranked_labels]

    return np.mean(is_in)  # type: ignore[no-any-return]


def scoring_neg_coverage(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """
    Supports multilabel classification.

    Evaluates how far we need, on average, to go down the list of labels in order to cover
    all the proper labels of the instance.

    - The ideal value is 1
    - The worst value is 0

    The result is equivalent to executing the following code:

    .. code-block:: python

        scores = np.array(scores)
        labels = np.array(labels)

        n_classes = scores.shape[1]
        from scipy.stats import rankdata
        int_ranks = rankdata(scores, axis=1)  # int ranks are from [1, n_classes]
        filtered_ranks = int_ranks * labels  # guarantee that 0 labels wont have max rank
        max_ranks = np.max(filtered_ranks, axis=1)
        float_ranks = (max_ranks - 1) / (n_classes - 1)  # float ranks are from [0,1]
        res = 1 - np.mean(float_ranks)

    :param labels: ground truth labels for each utterance
    :param scores: for each utterance, this list contains scores for each of `n_classes` classes
    :return: Score of the scoring metric
    """
    labels_, scores_ = transform(labels, scores)

    n_classes = scores_.shape[1]
    return 1 - (coverage_error(labels, scores) - 1) / (n_classes - 1)  # type: ignore[no-any-return]


def scoring_neg_ranking_loss(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """
    supports multilabel.

    Compute the average number of label pairs that are incorrectly ordered given y_score
    weighted by the size of the label set and the number of labels not in the label set.

    the ideal value is 0

    :param labels: ground truth labels for each utterance
    :param scores: for each utterance, this list contains scores for each of `n_classes` classes
    :return: Score of the scoring metric
    """
    return -label_ranking_loss(labels, scores)  # type: ignore[no-any-return]


def scoring_map(labels: LABELS_VALUE_TYPE, scores: SCORES_VALUE_TYPE) -> float:
    """
    Supports multilabel.

    mean average precision score

    the ideal value is 1, the worst is 0


    :param labels: ground truth labels for each utterance
    :param scores: for each utterance, this list contains scores for each of `n_classes` classes
    :return: Score of the scoring metric
    """
    return label_ranking_average_precision_score(labels, scores)  # type: ignore[no-any-return]
