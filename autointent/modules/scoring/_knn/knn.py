"""KNNScorer class for k-nearest neighbors scoring."""

from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import PositiveInt

from autointent import Context, VectorIndex
from autointent.custom_types import WEIGHT_TYPES, ListOfLabels
from autointent.modules.abc import ScoringModule
from autointent.schemas import EmbedderConfig

from .weighting import apply_weights


class KNNScorer(ScoringModule):
    """
    K-nearest neighbors (KNN) scorer for intent classification.

    This module uses a vector index to retrieve nearest neighbors for query utterances
    and applies a weighting strategy to compute class probabilities.

    :ivar weights: Weighting strategy used for scoring.
    :ivar _vector_index: VectorIndex instance for neighbor retrieval.
    :ivar name: Name of the scorer, defaults to "knn".

    Examples
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
        print(probabilities)  # Outputs predicted class probabilities for the utterances

    .. testoutput::

        [[0.67297815 0.32702185]
         [0.44031667 0.55968333]]

    """

    _vector_index: VectorIndex
    name = "knn"
    _n_classes: int
    _multilabel: bool
    supports_multilabel = True
    supports_multiclass = True

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any],
        k: PositiveInt,
        weights: WEIGHT_TYPES = "distance",
    ) -> None:
        """
        Initialize the KNNScorer.

        :param embedder_config: Config of the embedder used for vectorization.
        :param k: Number of closest neighbors to consider during inference.
        :param weights: Weighting strategy:
            - "uniform": Equal weight for all neighbors.
            - "distance": Weight inversely proportional to distance.
            - "closest": Only the closest neighbor of each class is weighted.
        """
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self.k = k
        self.weights = weights

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: PositiveInt,
        weights: WEIGHT_TYPES,
        embedder_config: EmbedderConfig | str | None = None,
    ) -> "KNNScorer":
        """
        Create a KNNScorer instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param k: Number of closest neighbors to consider during inference.
        :param weights: Weighting strategy for scoring.
        :param embedder_config: Config of the embedder, or None to use the best embedder.
        :return: Initialized KNNScorer instance.
        """
        if embedder_config is None:
            embedder_config = context.optimization_info.get_best_embedder()

        return cls(
            embedder_config=embedder_config,
            k=k,
            weights=weights,
        )

    def get_embedder_config(self) -> dict[str, Any]:
        """
        Get the name of the embedder.

        :return: Embedder name.
        """
        return self.embedder_config.model_dump()

    def fit(self, utterances: list[str], labels: ListOfLabels, clear_cache: bool = False) -> None:
        """
        Fit the scorer by training or loading the vector index.

        :param utterances: List of training utterances.
        :param labels: List of labels corresponding to the utterances.
        :raises ValueError: If the vector index mismatches the provided utterances.
        """
        if hasattr(self, "_vector_index") and clear_cache:
            self.clear_cache()

        self._validate_task(labels)

        self._vector_index = VectorIndex(self.embedder_config)
        self._vector_index.add(utterances, labels)

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """
        Predict class probabilities for the given utterances.

        :param utterances: List of query utterances.
        :return: Array of predicted probabilities for each class.
        """
        return self._predict(utterances)[0]

    def predict_with_metadata(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[dict[str, Any]] | None]:
        """
        Predict class probabilities along with metadata for the given utterances.

        :param utterances: List of query utterances.
        :return: Tuple of predicted probabilities and metadata with neighbor information.
        """
        scores, neighbors = self._predict(utterances)
        metadata = [{"neighbors": utterance_neighbors} for utterance_neighbors in neighbors]
        return scores, metadata

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the vector index."""
        self._vector_index.clear_ram()

    def _get_neighbours(self, utterances: list[str]) -> tuple[list[ListOfLabels], list[list[float]], list[list[str]]]:
        return self._vector_index.query(utterances, self.k)

    def _count_scores(self, labels: npt.NDArray[Any], distances: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return apply_weights(labels, distances, self.weights, self._n_classes, self._multilabel)

    def _predict(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[list[str]]]:
        """
        Predict class probabilities and retrieve neighbors for the given utterances.

        :param utterances: List of query utterances.
        :return: Tuple containing class probabilities and neighbor utterances.
        """
        labels, distances, neighbors = self._get_neighbours(utterances)
        scores = self._count_scores(np.array(labels), np.array(distances))
        return scores, neighbors
