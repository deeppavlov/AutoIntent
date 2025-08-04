"""DescriptionScorer classes for scoring utterances based on intent descriptions."""

from abc import ABC, abstractmethod

import numpy as np
import scipy
from numpy.typing import NDArray
from pydantic import PositiveFloat

from autointent import Context
from autointent.context.optimization_info import ScorerArtifact
from autointent.custom_types import ListOfLabels
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL
from autointent.modules.base import BaseScorer


class BaseDescriptionScorer(BaseScorer, ABC):
    """Base class for description-based scorers.

    This abstract base class defines the common interface and functionality
    for both bi-encoder and cross-encoder description scorers.

    Args:
        temperature: Temperature parameter for scaling logits, defaults to 1.0
        multilabe: Flag indicating classification task type
    """

    supports_multiclass = True
    supports_multilabel = True

    def __init__(self, temperature: PositiveFloat = 1.0, multilabel: bool = False) -> None:
        self.temperature = temperature
        self._multilabel = multilabel
        self._validate_temperature()

    def _validate_temperature(self) -> None:
        """Validate the temperature parameter."""
        if self.temperature < 0 or not isinstance(self.temperature, float | int):
            msg = "`temperature` argument must be a positive float"
            raise ValueError(msg)

    def _validate_descriptions(self, descriptions: list[str]) -> None:
        """Validate that descriptions don't contain None values.

        Args:
            descriptions: List of intent descriptions to validate

        Raises:
            ValueError: If descriptions contain None values
        """
        if any(description is None for description in descriptions):
            error_text = (
                "Some intent descriptions (label_description) are missing (None). "
                "Please ensure all intents have descriptions."
            )
            raise ValueError(error_text)

    def _apply_temperature_scaling(self, similarities: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply temperature scaling to convert similarities to probabilities.

        Args:
            similarities: Raw similarity scores

        Returns:
            Temperature-scaled probabilities
        """
        if self._multilabel:
            return scipy.special.expit(similarities / self.temperature)  # type: ignore[no-any-return]
        return scipy.special.softmax(similarities / self.temperature, axis=1)  # type: ignore[no-any-return]

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
        descriptions: list[str],
    ) -> None:
        """Fit the scorer by processing utterances and descriptions.

        Args:
            utterances: List of utterances to process
            labels: List of labels corresponding to the utterances
            descriptions: List of intent descriptions

        Raises:
            ValueError: If descriptions contain None values
        """
        self._validate_descriptions(descriptions)
        self._fit_implementation(descriptions)

    @abstractmethod
    def _fit_implementation(self, descriptions: list[str]) -> None:
        """Implementation-specific fitting logic.

        Args:
            descriptions: List of intent descriptions
        """

    def predict(self, utterances: list[str]) -> NDArray[np.float64]:
        """Predict scores for utterances based on similarity to intent descriptions.

        Args:
            utterances: List of utterances to score

        Returns:
            Array of probabilities for each utterance
        """
        similarities = self._compute_similarities(utterances)
        return self._apply_temperature_scaling(similarities)

    @abstractmethod
    def _compute_similarities(self, utterances: list[str]) -> NDArray[np.float64]:
        """Compute similarity scores between utterances and descriptions.

        Args:
            utterances: List of utterances to score

        Returns:
            Array of similarity scores
        """

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear cached data in memory."""

    def get_train_data(self, context: Context) -> tuple[list[str], ListOfLabels, list[str]]:
        """Get training data from context.

        Args:
            context: Context containing training data

        Returns:
            Tuple containing utterances, labels, and descriptions
        """
        return (  # type: ignore[return-value]
            context.data_handler.train_utterances(0),
            context.data_handler.train_labels(0),
            context.data_handler.intent_descriptions,
        )

    def score_cv(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """Evaluate the scorer on a test set and compute the specified metrics.

        Args:
            context: Context containing test set and other data
            metrics: List of metric names to compute

        Returns:
            Dictionary of computed metric values
        """
        metrics_dict = SCORING_METRICS_MULTILABEL if context.is_multilabel() else SCORING_METRICS_MULTICLASS
        chosen_metrics = {name: fn for name, fn in metrics_dict.items() if name in metrics}

        metrics_calculated, all_val_scores = self.score_metrics_cv(
            chosen_metrics,
            context.data_handler.validation_iterator(),
            descriptions=context.data_handler.intent_descriptions,
        )

        self._artifact = ScorerArtifact(folded_scores=all_val_scores)

        return metrics_calculated
