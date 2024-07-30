import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def prediction_accuracy(y_true: list[int], y_pred: list[int]):
    indicators = np.array(y_true) == np.array(y_pred)
    return np.mean(indicators)


def prediction_roc_auc(y_true: list[int], y_pred: list[int]):
    return roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")


def prediction_precision(y_true: list[int], y_pred: list[int]):
    return precision_score(y_true, y_pred, average="macro")


def prediction_recall(y_true: list[int], y_pred: list[int]):
    return recall_score(y_true, y_pred, average="macro")


def prediction_f1(y_true: list[int], y_pred: list[int]):
    return f1_score(y_true, y_pred, average="macro")
