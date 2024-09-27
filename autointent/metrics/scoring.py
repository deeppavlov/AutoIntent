import warnings
from typing import Protocol

import numpy as np
from sklearn.metrics import roc_auc_score, coverage_error, label_ranking_loss, label_ranking_average_precision_score

from .prediction import (
    prediction_accuracy,
    prediction_f1,
    prediction_precision,
    prediction_recall,
)


class ScoringMetricFn(Protocol):
    def __call__(self, labels: list[int] | list[list[int]], scores: list[list[float]]) -> float:
        """
        Arguments
        ---
        - `labels`: ground truth labels for each utterance
            - multiclass case: list representing an array of shape `(n_samples,)` with integer values
            - multilabel case: list representing a matrix of shape `(n_samples, n_classes)` with integer values
        - `scores`: for each utterance, this list contains scores for each of `n_classes` classes
        """
        ...


def scoring_log_likelihood(labels: list[int] | list[list[int]], scores: list[list[float]]) -> float:
    """
    TODO test multilabel case

    supports multiclass and multilabel

    Multiclass case
    ---
    mean negative cross-entropy for each utterance classification result, i.e.
    ```math
    {1\\over\\ell}\\sum_{i=1}^\\ell log(s[y[i]]),
    ```
    where `s[y[i]]` is a predicted score of `i`th utterance having ground truth label

    Multilabel case
    ---
    mean negative binary cross-entropy, i.e.
    ```math
    {1\\over\\ell}\\sum_{i=1}^\\ell\\sum_{c=1}^C [y[i,c]\\cdot\\log(s[i,c])+(1-y[i,c])\\cdot\\log(1-s[i,c])]
    ```
    where `s[i,c]` is a predicted score of `i`th utterance having ground truth label `c`
    """
    scores_array = np.array(scores)
    labels_array = np.array(labels)

    if np.any((scores_array <= 0) | (scores_array > 1)):
        warnings.warn("One or more scores are not from [0,1]")

    if labels_array.ndim == 1:
        relevant_scores = scores_array[np.arange(len(labels_array)), labels_array]
        res = np.mean(np.log(relevant_scores).clip(min=-100, max=100))
    else:
        log_likelihood = labels_array * np.log(scores_array) + (1 - labels_array) * np.log(1 - scores_array)
        clipped_one = log_likelihood.clip(min=-100, max=100)
        res = clipped_one.mean()
    return res


def scoring_roc_auc(labels: list[int] | list[list[int]], scores: list[list[float]]) -> float:
    """
    supports multiclass and multilabel

    macro averaged roc-auc for utterance classification task, i.e.
    ```math
    {1\\over C}\\sum_{k=1}^C ROCAUC(scores[:, k], labels[:, k])
    ```
    """
    scores = np.array(scores)
    labels = np.array(labels)

    n_classes = scores.shape[1]
    if labels.ndim == 1:
        labels = (labels[:, None] == np.arange(n_classes)[None, :]).astype(int)

    return roc_auc_score(labels, scores, average="macro")


def scoring_accuracy(labels: list[int] | list[list[int]], scores: list[list[float]]) -> float:
    """
    supports multiclass and multilabel
    """
    scores = np.array(scores)
    labels = np.array(labels)

    if labels.ndim == 1:
        pred_labels = np.argmax(scores, axis=1)
        res = prediction_accuracy(labels, pred_labels)
    else:
        pred_labels = (scores > 0.5).astype(int)
        res = prediction_accuracy(labels, pred_labels)

    return res


def scoring_f1(labels: list[int] | list[list[int]], scores: list[list[float]]) -> float:
    """
    supports multiclass and multilabel
    """
    scores = np.array(scores)
    labels = np.array(labels)

    if labels.ndim == 1:
        pred_labels = np.argmax(scores, axis=1)
        res = prediction_f1(labels, pred_labels)
    else:
        pred_labels = (scores > 0.5).astype(int)
        res = prediction_f1(labels, pred_labels)

    return res


def scoring_precision(labels: list[int] | list[list[int]], scores: list[list[float]]) -> float:
    """
    supports multiclass and multilabel
    """
    scores = np.array(scores)
    labels = np.array(labels)

    if labels.ndim == 1:
        pred_labels = np.argmax(scores, axis=1)
        res = prediction_precision(labels, pred_labels)
    else:
        pred_labels = (scores > 0.5).astype(int)
        res = prediction_precision(labels, pred_labels)

    return res


def scoring_recall(labels: list[int] | list[list[int]], scores: list[list[float]]) -> float:
    """
    supports multiclass and multilabel
    """
    scores = np.array(scores)
    labels = np.array(labels)

    if labels.ndim == 1:
        pred_labels = np.argmax(scores, axis=1)
        res = prediction_recall(labels, pred_labels)
    else:
        pred_labels = (scores > 0.5).astype(int)
        res = prediction_recall(labels, pred_labels)

    return res


def scoring_hit_rate(labels: list[list[int]], scores: list[list[float]]):
    """
    TODO test this function

    supports multilabel

    calculates fraction of cases when the top-ranked label is in the set of proper labels of the instance
    """
    scores = np.array(scores)
    labels = np.array(labels)

    top_ranked_labels = np.argmax(scores, axis=1)
    is_in = labels[np.arange(len(labels)), top_ranked_labels]

    return np.mean(is_in)


def scoring_neg_coverage(labels: list[list[int]], scores: list[list[float]]):
    """
    TODO test this function

    supports multilabel

    evaluates how far we need, on the average, to go down the list of labels
    in order to cover all the proper labels of the instance

    the ideal value is 1, the worst is 0
    """
    # scores = np.array(scores)
    # labels = np.array(labels)

    # n_classes = scores.shape[1]
    # from scipy.stats import rankdata
    # int_ranks = rankdata(scores, axis=1)  # int ranks are from [1, n_classes]
    # filtered_ranks = int_ranks * labels  # guarantee that 0 labels wont have max rank
    # max_ranks = np.max(filtered_ranks, axis=1)
    # float_ranks = (max_ranks - 1) / (n_classes - 1)  # float ranks are from [0,1]
    # res = 1 - np.mean(float_ranks)

    n_classes = len(labels[0])
    res = 1 - (coverage_error(labels, scores) - 1) / (n_classes - 1)

    return res


def scoring_neg_ranking_loss(labels: list[list[int]], scores: list[list[float]]):
    """
    supports multilabel

    Compute the average number of label pairs that are incorrectly ordered given y_score
    weighted by the size of the label set and the number of labels not in the label set.

    the ideal value is 0
    """
    return -label_ranking_loss(labels, scores)


def scoring_map(labels: list[list[int]], scores: list[list[float]]):
    """
    supports multilabel

    mean average precision score

    the ideal value is 1, the worst is 0
    """
    return label_ranking_average_precision_score(labels, scores)
