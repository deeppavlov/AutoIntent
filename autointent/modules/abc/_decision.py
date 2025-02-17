"""Predictior module."""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from autointent import Context
from autointent.context.optimization_info import DecisionArtifact
from autointent.custom_types import ListOfGenericLabels
from autointent.metrics import DECISION_METRICS
from autointent.modules.abc import BaseModule
from autointent.schemas import Tag


class BaseDecision(BaseModule, ABC):
    """Base class for decision modules."""

    @abstractmethod
    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: ListOfGenericLabels,
        tags: list[Tag] | None = None,
    ) -> None:
        """
        Fit the model.

        :param scores: Scores to fit
        :param labels: Labels to fit
        :param tags: Tags to fit
        """

    @abstractmethod
    def predict(self, scores: npt.NDArray[Any]) -> ListOfGenericLabels:
        """
        Predict the best score.

        :param scores: Scores to predict
        """

    def score_ho(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """
        Calculate metric on test set and return metric value.

        :param context: Context to score
        :param split: Target split
        :return: Computed metrics value for the test set or error code of metrics
        """
        train_scores, train_labels, tags = self.get_train_data(context)
        self.fit(train_scores, train_labels, tags)

        val_labels, val_scores = get_decision_evaluation_data(context, "validation")
        decisions = self.predict(val_scores)
        chosen_metrics = {name: fn for name, fn in DECISION_METRICS.items() if name in metrics}
        self._artifact = DecisionArtifact(labels=decisions)
        return self.score_metrics_ho((val_labels, decisions), chosen_metrics)

    def score_cv(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """
        Calculate metric on test set and return metric value.

        :param context: Context to score
        :param split: Target split
        :return: Computed metrics value for the test set or error code of metrics
        """
        labels = context.data_handler.train_labels_folded()
        scores = context.optimization_info.get_best_folded_scores()

        if scores is None:
            msg = "No folded scores are found."
            raise RuntimeError(msg)

        chosen_metrics = {name: fn for name, fn in DECISION_METRICS.items() if name in metrics}
        metrics_values: dict[str, list[float]] = {name: [] for name in chosen_metrics}
        all_val_decisions = []
        for j in range(context.data_handler.n_folds):
            val_labels = labels[j]
            val_scores = scores[j]
            train_folds = [i for i in range(context.data_handler.n_folds) if i != j]
            train_labels = [ut for i_fold in train_folds for ut in labels[i_fold]]
            train_scores = np.array([sc for i_fold in train_folds for sc in scores[i_fold]])
            self.fit(train_scores, train_labels, context.data_handler.tags)  # type: ignore[arg-type]
            val_decisions = self.predict(val_scores)
            for name, fn in chosen_metrics.items():
                metrics_values[name].append(fn(val_labels, val_decisions))
            all_val_decisions.append(val_decisions)

        self._artifact = DecisionArtifact(labels=[pred for pred_list in all_val_decisions for pred in pred_list])
        return {name: float(np.mean(values_list)) for name, values_list in metrics_values.items()}

    def get_assets(self) -> DecisionArtifact:
        """Return useful assets that represent intermediate data into context."""
        return self._artifact

    def clear_cache(self) -> None:
        """Clear cache."""

    def _validate_task(self, scores: npt.NDArray[Any], labels: ListOfGenericLabels) -> None:
        self._n_classes, self._multilabel, self._oos = self._get_task_specs(labels)
        self._validate_multilabel(self._multilabel)
        self._validate_oos(self._oos, raise_error=False)
        if self._n_classes != scores.shape[1]:
            msg = (
                "There is a mismatch between provided labels and scores. "
                f"Labels contain {self._n_classes} classes, but scores contain "
                f"probabilities for {scores.shape[1]} classes."
            )
            raise ValueError(msg)

    def get_train_data(self, context: Context) -> tuple[npt.NDArray[Any], ListOfGenericLabels, list[Tag]]:
        labels, scores = get_decision_evaluation_data(context, "train")
        return (scores, labels, context.data_handler.tags)


def get_decision_evaluation_data(
    context: Context,
    split: Literal["train", "validation"],
) -> tuple[ListOfGenericLabels, npt.NDArray[np.float64]]:
    """
    Get decision evaluation data.

    :param context: Context
    :param split: Target split
    :return:
    """
    if split == "train":
        labels = context.data_handler.train_labels(1)
        scores = context.optimization_info.get_best_train_scores()
    elif split == "validation":
        labels = context.data_handler.validation_labels(1)
        scores = context.optimization_info.get_best_validation_scores()
    else:
        message = f"Invalid split '{split}' provided. Expected one of 'train', 'validation'."
        raise ValueError(message)

    if scores is None:
        message = f"No '{split}' scores found in the optimization info"
        raise ValueError(message)

    return labels, scores
