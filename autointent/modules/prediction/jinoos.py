from warnings import warn

import numpy as np

from .base import Context, PredictionModule, get_prediction_evaluation_data


class JinoosPredictor(PredictionModule):
    default_search_space = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def __init__(self, search_space: list[float] = None):
        self.search_space = search_space if search_space is not None else self.default_search_space

    def fit(self, context: Context):
        """
        TODO: use dev split instead of test split
        """

        if not context.data_handler.has_oos_samples():
            warn(
                "Your data doesn't contain out-of-scope utterances."
                "Using JinoosPredictor imposes unnecessary computational overhead."
            )

        y_true, scores = get_prediction_evaluation_data(context)
        pred_classes, best_scores = _predict(scores)

        metrics_list = []
        for thresh in self.search_space:
            y_pred = _detect_oos(pred_classes, best_scores, thresh)
            metric_value = jinoos_score(y_true, y_pred)
            metrics_list.append(metric_value)

        self._thresh = self.search_space[np.argmax(metrics_list)]

    def predict(self, scores: list[list[float]]):
        pred_classes, best_scores = _predict(scores)
        return _detect_oos(pred_classes, best_scores, self._thresh)


def _predict(scores):
    pred_classes = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(len(scores)), pred_classes]
    return pred_classes, best_scores


def _detect_oos(classes, scores, thresh):
    classes[scores < thresh] = -1  # out of scope
    return classes


def jinoos_score(y_true: list[int], y_pred: list[int]):
    """
    joint in and out of scope score

    ```math
    \\frac{C_{in}}{N_{in}}+\\frac{C_{oos}}{N_{oos}},
    ```

    where $C_{in}$ is the number of correctly predicted in-domain labels, and $N_{in}$ is the total number of in-domain labels. The same for OOS samples
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    in_domain_mask = y_true != -1
    correct_mask = y_true == y_pred

    correct_in_domain = np.sum(correct_mask & in_domain_mask)
    total_in_domain = np.sum(in_domain_mask)
    accuracy_in_domain = correct_in_domain / total_in_domain

    correct_oos = np.sum(correct_mask & ~in_domain_mask)
    total_oos = np.sum(~in_domain_mask)
    accuracy_oos = correct_oos / total_oos

    return accuracy_in_domain + accuracy_oos
