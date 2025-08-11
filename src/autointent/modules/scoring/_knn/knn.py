"""KNNScorer class for k-nearest neighbors scoring."""

from typing import Any, get_args

import numpy as np
import numpy.typing as npt
from pydantic import PositiveInt

from autointent import Context, VectorIndex
from autointent.configs import EmbedderConfig
from autointent.custom_types import ListOfLabels, WeightType
from autointent.modules.base import BaseScorer

from .weighting import apply_weights


class KNNScorer(BaseScorer):
    """K-nearest neighbors (KNN) scorer for intent classification.

    This module uses a vector index to retrieve nearest neighbors for query utterances
    and applies a weighting strategy to compute class probabilities.

    Args:
        embedder_config: Config of the embedder used for vectorization
        k: Number of closest neighbors to consider during inference
        weights: Weighting strategy:

            - "uniform": Equal weight for all neighbors
            - "distance": Weight inversely proportional to distance
            - "closest": Only the closest neighbor of each class is weighted

    Examples:
    --------

    .. testcode::

        from autointent.modules.scoring import KNNScorer
        utterances = ["hello", "how are you?"]
        labels = [0, 1]
        scorer = KNNScorer(
            embedder_config="sergeyzh/rubert-tiny-turbo",
            k=5,
        )
        scorer.fit(utterances, labels)
        test_utterances = ["hi", "what's up?"]
        probabilities = scorer.predict(test_utterances)

    """

    _vector_index: VectorIndex
    name = "knn"
    _n_classes: int
    _multilabel: bool
    supports_multilabel = True
    supports_multiclass = True

    def __init__(
        self,
        k: PositiveInt = 5,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        weights: WeightType = "distance",
    ) -> None:
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self.k = k
        self.weights = weights

        if self.k < 0 or not isinstance(self.k, int):
            msg = "`k` argument of `KNNScorer` must be a positive int"
            raise ValueError(msg)

        if weights not in get_args(WeightType):
            msg = f"`weights` argument of `KNNScorer` must be a literal from a list: {get_args(WeightType)}"
            raise TypeError(msg)

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: PositiveInt = 5,
        weights: WeightType = "distance",
        embedder_config: EmbedderConfig | str | None = None,
    ) -> "KNNScorer":
        """Create a KNNScorer instance using a Context object.

        Args:
            context: Context containing configurations and utilities
            k: Number of closest neighbors to consider during inference
            weights: Weighting strategy for scoring
            embedder_config: Config of the embedder, or None to use the best embedder
        """
        if embedder_config is None:
            embedder_config = context.resolve_embedder()

        return cls(
            embedder_config=embedder_config,
            k=k,
            weights=weights,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {"embedder_config": self.embedder_config.model_dump()}

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        """Fit the scorer by training or loading the vector index.

        Args:
            utterances: List of training utterances
            labels: List of labels corresponding to the utterances
            clear_cache: Whether to clear the vector index cache before fitting

        Raises:
            ValueError: If the vector index mismatches the provided utterances
        """
        self._validate_task(labels)

        self._vector_index = VectorIndex(self.embedder_config)
        self._vector_index.add(utterances, labels)

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """Predict class probabilities for the given utterances.

        Args:
            utterances: List of query utterances

        Returns:
            Array of predicted probabilities for each class
        """
        return self._predict(utterances)[0]

    def predict_with_metadata(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[dict[str, Any]] | None]:
        """Predict class probabilities along with metadata for the given utterances.

        Args:
            utterances: List of query utterances

        Returns:
            Tuple containing:
                - Array of predicted probabilities
                - List of metadata with neighbor information
        """
        scores, neighbors = self._predict(utterances)
        metadata = [{"neighbors": utterance_neighbors} for utterance_neighbors in neighbors]
        return scores, metadata

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the vector index."""
        if hasattr(self, "_vector_index"):
            self._vector_index.clear_ram()

    def _get_neighbours(self, utterances: list[str]) -> tuple[list[ListOfLabels], list[list[float]], list[list[str]]]:
        """Get nearest neighbors for given utterances.

        Args:
            utterances: List of query utterances

        Returns:
            Tuple containing:
                - List of labels for neighbors
                - List of distances to neighbors
                - List of neighbor utterances
        """
        return self._vector_index.query(utterances, self.k)

    def _count_scores(self, labels: npt.NDArray[Any], distances: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Calculate weighted scores for labels based on distances.

        Args:
            labels: Array of neighbor labels
            distances: Array of distances to neighbors

        Returns:
            Array of weighted scores
        """
        return apply_weights(labels, distances, self.weights, self._n_classes, self._multilabel)

    def _predict(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[list[str]]]:
        """Predict class probabilities and retrieve neighbors for the given utterances.

        Args:
            utterances: List of query utterances

        Returns:
            Tuple containing:
                - Array of class probabilities
                - List of neighbor utterances
        """
        labels, distances, neighbors = self._get_neighbours(utterances)
        scores = self._count_scores(np.array(labels), np.array(distances))
        return scores, neighbors
