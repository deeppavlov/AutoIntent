from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.context.data_handler import Tag
from autointent.context.optimization_info import PredictorArtifact
from autointent.custom_types import LabelType
from autointent.metrics import PredictionMetricFn
from autointent.modules.base import Module


class PredictionModule(Module, ABC):
    """Base class for prediction modules."""

    @abstractmethod
    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: list[LabelType],
        tags: list[Tag] | None = None,
    ) -> None:
        """
        Fit the model.

        :param scores: Scores to fit
        :param labels: Labels to fit
        :param tags: Tags to fit
        :return:
        """

    @abstractmethod
    def predict(self, scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Predict the best score.

        :param scores: Scores to predict
        :return:
        """

    def score(self, context: Context, metric_fn: PredictionMetricFn) -> float:
        """
        Calculate metric on test set and return metric value.

        :param context: Context to score
        :param metric_fn: Metric function
        :return: Score
        """
        labels, scores = get_prediction_evaluation_data(context)
        self._predictions = self.predict(scores)
        return metric_fn(labels, self._predictions)

    def get_assets(self) -> PredictorArtifact:
        """
        Return useful assets that represent intermediate data into context.

        :return:
        """
        return PredictorArtifact(labels=self._predictions)

    def clear_cache(self) -> None:
        """Clear cache."""


def get_prediction_evaluation_data(
    context: Context,
) -> tuple[list[LabelType], npt.NDArray[Any]]:
    """
    Get prediction evaluation data.
    :param context: Context
    :return:
    """
    labels = np.array(context.data_handler.labels_test)
    scores = context.optimization_info.get_best_test_scores()

    if scores is None:
        msg = "No test scores found in the optimization info"
        raise ValueError(msg)

    oos_scores = context.optimization_info.get_best_oos_scores()
    return_scores = scores
    if oos_scores is not None:
        oos_labels = (
            [[0] * context.get_n_classes()] * len(oos_scores) if context.is_multilabel() else [-1] * len(oos_scores)  # type: ignore[list-item]
        )
        labels = np.concatenate([labels, np.array(oos_labels)])
        return_scores = np.concatenate([scores, oos_scores])

    return labels.tolist(), return_scores


def apply_tags(labels: npt.NDArray[Any], scores: npt.NDArray[Any], tags: list[Tag]) -> npt.NDArray[Any]:
    """
    this function is intended to be used with multilabel predictor.

    If some intent classes have common tag (i.e. they are mutually exclusive)
    and were assigned to one sample, leave only that class that has the highest score.

    :param labels: Binary labels of shape (n_samples, n_classes)
    :param scores: (n_samples, n_classes) with float values from 0..1
    :param tags: list of Tags
    :return: np.ndarray of shape (n_samples, n_classes) with binary labels
    """
    n_samples, _ = labels.shape
    res = np.copy(labels)

    for i in range(n_samples):
        sample_labels = labels[i].astype(bool)
        sample_scores = scores[i]

        for tag in tags:
            if any(sample_labels[idx] for idx in tag.intent_ids):
                # Find the index of the class with the highest score among the tagged indices
                max_score_index = max(tag.intent_ids, key=lambda idx: sample_scores[idx])
                # Set all other tagged indices to 0 in the res
                for idx in tag.intent_ids:
                    if idx != max_score_index:
                        res[i, idx] = 0

    return res
