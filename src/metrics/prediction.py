import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def prediction_accuracy(y_true: list[int], y_pred: list[int]):
    indicators = np.array(y_true) == np.array(y_pred)
    return np.mean(indicators)


def prediction_roc_auc(y_true: list[int], y_pred: list[int]):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    n_classes = len(np.unique(y_true))

    roc_auc_scores = []
    for k in range(n_classes):
        binarized_true = (y_true == k).astype(int)
        binarized_pred = (y_pred == k).astype(int)
        roc_auc = roc_auc_score(binarized_true, binarized_pred)
        roc_auc_scores.append(roc_auc)

    macro_roc_auc = np.mean(roc_auc_scores)

    return macro_roc_auc


def prediction_precision(y_true: list[int], y_pred: list[int]):
    return precision_score(y_true, y_pred, average="macro")


def prediction_recall(y_true: list[int], y_pred: list[int]):
    return recall_score(y_true, y_pred, average="macro")


def prediction_f1(y_true: list[int], y_pred: list[int]):
    return f1_score(y_true, y_pred, average="macro")
