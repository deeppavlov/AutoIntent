def regexp_partial_accuracy(y_true: list[int], y_pred: list[set[int]]):
    correct = sum(true in pred for true, pred in zip(y_true, y_pred, strict=False))
    total = len(y_true)
    if total == 0:
        return -1  # TODO think about it
    return correct / total


def regexp_partial_precision(y_true: list[int], y_pred: list[set[int]]):
    correct = sum(true in pred for true, pred in zip(y_true, y_pred, strict=False))
    total = sum(len(pred) > 0 for pred in y_pred)
    if total == 0:
        return -1
    return correct / total
