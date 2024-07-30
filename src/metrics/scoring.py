import numpy as np
from sklearn.metrics import roc_auc_score


def scoring_neg_cross_entropy(scores: list[list[float]], labels: list[int]) -> float:
    """
    Arguments
    ---
    `scores`: for each utterance, this list contains scores for each of `n_classes` classes
    `labels`: ground truth labels for each utterance

    Return
    ---
    mean negative cross-entropy for each utterance classification result, i.e.
    ```math
    {1\over\ell}\sum_{i=1}^\ell-log(s[y[i]]),
    ```
    where s[y[i]] is a predicted score of ith utterance having ground truth label
    """
    scores_array = np.array(scores)
    labels_array = np.array(labels)

    relevant_scores = scores_array[np.arange(len(labels_array)), labels_array]

    if np.any((relevant_scores <= 0) | (relevant_scores > 1)):
        raise ValueError("One or more scores are non-positive")

    return np.mean(-np.log(relevant_scores))


def scoring_roc_auc(scores: list[list[float]], labels: list[int]) -> float:
    """
    Arguments
    ---
    `scores`: for each utterance, this list contains scores for each of `n_classes` classes
    `labels`: ground truth labels for each utterance

    Return
    ---
    macro averaged roc-auc for utterance classification task, i.e.
    ```math
    {1\over C}\sum_{k=1}^C ROCAUC(scores[:, k], labels[:, k])
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
