from typing import Protocol


class RegexpMetricFn(Protocol):
    def __call__(self, y_true: list[int], y_pred: list[list[int]]) -> float: ...


def regexp_partial_accuracy(y_true: list[int], y_pred: list[list[int]]) -> float:
    correct = sum(true in pred for true, pred in zip(y_true, y_pred, strict=True))
    total = len(y_true)
    if total == 0:
        return -1  # TODO think about it
    return correct / total


def regexp_partial_precision(y_true: list[int], y_pred: list[list[int]]) -> float:
    correct = sum(true in pred for true, pred in zip(y_true, y_pred, strict=True))
    total = sum(len(pred) > 0 for pred in y_pred)
    if total == 0:
        return -1
    return correct / total
