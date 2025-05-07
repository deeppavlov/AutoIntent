"""Predictor module."""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import assert_never

from autointent import Context
from autointent.context.optimization_info import DecisionArtifact
from autointent.custom_types import ListOfGenericLabels
from autointent.metrics import DECISION_METRICS
from autointent.modules.base import BaseModule
from autointent.schemas import Tag


class BaseDecision(BaseModule, ABC):
    """Base class for decision modules."""

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def fit(
        self,
        scores: npt.NDArray[Any],
        labels: ListOfGenericLabels,
        tags: list[Tag] | None = None,
    ) -> None:
        """Fit the model.

        Args:
            scores: Scores to fit
            labels: Labels to fit
            tags: Tags to fit
        """

    @abstractmethod
    def predict(self, scores: npt.NDArray[Any]) -> ListOfGenericLabels:
        """Predict the best score.

        Args:
            scores: Scores to predict

        Returns:
            Predicted labels
        """

    def score_ho(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """Calculate metric on test set and return metric value.

        Args:
            context: Context to score
            metrics: List of metrics to compute

        Returns:
            Dictionary of computed metrics values for the test set

        Raises:
            RuntimeError: If no folded scores are found
        """
        train_scores, train_labels, tags = self.get_train_data(context)
        self.fit(train_scores, train_labels, tags)

        val_labels, val_scores = get_decision_evaluation_data(context, "validation")
        decisions = self.predict(val_scores)
        chosen_metrics = {name: fn for name, fn in DECISION_METRICS.items() if name in metrics}
        self._artifact = DecisionArtifact(labels=decisions)
        return self.score_metrics_ho((val_labels, decisions), chosen_metrics)

    def score_cv(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """Calculate metric on test set and return metric value.

        Args:
            context: Context to score
            metrics: List of metrics to compute

        Returns:
            Dictionary of computed metrics values for the test set

        Raises:
            RuntimeError: If no folded scores are found
        """
        labels = context.data_handler.train_labels_folded()
        scores = context.optimization_info.get_best_folded_scores()

        if scores is None:
            msg = "No folded scores are found."
            raise RuntimeError(msg)

        chosen_metrics = {name: fn for name, fn in DECISION_METRICS.items() if name in metrics}
        metrics_values: dict[str, list[float]] = {name: [] for name in chosen_metrics}
        all_val_decisions = []
        for j in range(context.data_handler.config.n_folds):
            val_labels = labels[j]
            val_scores = scores[j]
            train_folds = [i for i in range(context.data_handler.config.n_folds) if i != j]
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
        """Return useful assets that represent intermediate data into context.

        Returns:
            Decision artifact containing intermediate data
        """
        return self._artifact

    def clear_cache(self) -> None:
        """Clear cache."""

    def _validate_task(self, scores: npt.NDArray[Any], labels: ListOfGenericLabels) -> None:
        """Validate task specifications.

        Args:
            scores: Input scores
            labels: Input labels

        Raises:
            ValueError: If there is a mismatch between provided labels and scores
        """
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
        """Get training data from context.

        Args:
            context: Context containing the data

        Returns:
            Tuple containing scores, labels, and tags
        """
        labels, scores = get_decision_evaluation_data(context, "train")
        return (scores, labels, context.data_handler.tags)


def get_decision_evaluation_data(
    context: Context,
    split: Literal["train", "validation"],
) -> tuple[ListOfGenericLabels, npt.NDArray[np.float64]]:
    """Get decision evaluation data.

    Args:
        context: Context containing the data
        split: Target split (either 'train' or 'validation')

    Returns:
        Tuple containing labels and scores for the specified split

    Raises:
        ValueError: If invalid split name is provided or no scores are found
    """
    if split == "train":
        labels = context.data_handler.train_labels(1)
        scores = context.optimization_info.get_best_train_scores()
    elif split == "validation":
        labels = context.data_handler.validation_labels(1)
        scores = context.optimization_info.get_best_validation_scores()
    else:
        assert_never(split)

    if scores is None:
        message = f"No '{split}' scores found in the optimization info"
        raise ValueError(message)

    return labels, scores
