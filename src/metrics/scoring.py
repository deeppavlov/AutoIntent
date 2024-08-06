import numpy as np
from sklearn.metrics import roc_auc_score
from .prediction import prediction_accuracy, prediction_f1, prediction_precision, prediction_recall


def scoring_neg_cross_entropy(labels: list[int], scores: list[list[float]]) -> float:
    """
    Arguments
    ---
    `scores`: for each utterance, this list contains scores for each of `n_classes` classes
    `labels`: ground truth labels for each utterance

    Return
    ---
    mean negative cross-entropy for each utterance classification result, i.e.
    ```math
    {1\\over\\ell}\\sum_{i=1}^\\ell-log(s[y[i]]),
    ```
    where s[y[i]] is a predicted score of ith utterance having ground truth label
    """
    scores_array = np.array(scores)
    labels_array = np.array(labels)

    relevant_scores = scores_array[np.arange(len(labels_array)), labels_array]

    if np.any((relevant_scores <= 0) | (relevant_scores > 1)):
        raise ValueError("One or more scores are non-positive")

    return np.mean(-np.log(relevant_scores))


def scoring_roc_auc(labels: list[int], scores: list[list[float]]) -> float:
    """
    Arguments
    ---
    `scores`: for each utterance, this list contains scores for each of `n_classes` classes
    `labels`: ground truth labels for each utterance

    Return
    ---
    macro averaged roc-auc for utterance classification task, i.e.
    ```math
    {1\\over C}\\sum_{k=1}^C ROCAUC(scores[:, k], labels[:, k])
    ```
    """
    scores = np.array(scores)
    labels = np.array(labels)

    n_classes = scores.shape[1]

    roc_auc_scores = []
    for k in range(n_classes):
        binarized_labels = (labels == k).astype(int)
        roc_auc = roc_auc_score(binarized_labels, scores[:, k])
        roc_auc_scores.append(roc_auc)

    macro_roc_auc = np.mean(roc_auc_scores)

    return macro_roc_auc


def scoring_accuracy(labels: list[int], scores: list[list[float]]) -> float:
    pred_labels = np.argmax(scores, axis=1)
    return prediction_accuracy(labels, pred_labels)


def scoring_f1(labels: list[int], scores: list[list[float]]) -> float:
    pred_labels = np.argmax(scores, axis=1)
    return prediction_f1(labels, pred_labels)


def scoring_precision(labels: list[int], scores: list[list[float]]) -> float:
    pred_labels = np.argmax(scores, axis=1)
    return prediction_precision(labels, pred_labels)


def scoring_recall(labels: list[int], scores: list[list[float]]) -> float:
    pred_labels = np.argmax(scores, axis=1)
    return prediction_recall(labels, pred_labels)
