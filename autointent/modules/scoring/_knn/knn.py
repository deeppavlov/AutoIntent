"""KNNScorer class for k-nearest neighbors scoring."""

from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context, VectorIndex
from autointent.custom_types import WEIGHT_TYPES, ListOfLabels
from autointent.modules.abc import ScoringModule

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
            embedder_name="sergeyzh/rubert-tiny-turbo",
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
        embedder_name: str,
        k: int,
        weights: WEIGHT_TYPES = "distance",
        embedder_device: str = "cpu",
        embedder_batch_size: int = 32,
        embedder_max_length: int | None = None,
        embedder_use_cache: bool = True,
    ) -> None:
        """
        Initialize the KNNScorer.

        :param embedder_name: Name of the embedder used for vectorization.
        :param k: Number of closest neighbors to consider during inference.
        :param weights: Weighting strategy:
            - "uniform" (or False): Equal weight for all neighbors.
            - "distance" (or True): Weight inversely proportional to distance.
            - "closest": Only the closest neighbor of each class is weighted.
        :param embedder_device: Device to run operations on, e.g., "cpu" or "cuda".
        :param embedder_batch_size: Batch size for embedding generation, defaults to 32.
        :param embedder_max_length: Maximum sequence length for embedding, or None for default.
        :param embedder_use_cache: Flag indicating whether to cache intermediate embeddings.
        """
        self.embedder_name = embedder_name
        self.k = k
        self.weights = weights
        self.embedder_device = embedder_device
        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_length = embedder_max_length
        self.embedder_use_cache = embedder_use_cache

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: int,
        weights: WEIGHT_TYPES,
        embedder_name: str | None = None,
    ) -> "KNNScorer":
        """
        Create a KNNScorer instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param k: Number of closest neighbors to consider during inference.
        :param weights: Weighting strategy for scoring.
        :param embedder_name: Name of the embedder, or None to use the best embedder.
        :return: Initialized KNNScorer instance.
        """
        if embedder_name is None:
            embedder_name = context.optimization_info.get_best_embedder()

        return cls(
            embedder_name=embedder_name,
            k=k,
            weights=weights,
            embedder_device=context.get_device(),
            embedder_batch_size=context.get_batch_size(),
            embedder_max_length=context.get_max_length(),
            embedder_use_cache=context.get_use_cache(),
        )

    def get_embedder_name(self) -> str:
        """
        Get the name of the embedder.

        :return: Embedder name.
        """
        return self.embedder_name

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

        self._vector_index = VectorIndex(
            self.embedder_name,
            self.embedder_device,
            self.embedder_batch_size,
            self.embedder_max_length,
            self.embedder_use_cache,
        )
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
