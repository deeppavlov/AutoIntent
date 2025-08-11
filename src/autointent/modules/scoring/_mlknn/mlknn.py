"""MLKnnScorer class for multi-label k-nearest neighbors classification."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import NonNegativeInt, PositiveFloat, PositiveInt
from typing_extensions import assert_never

from autointent import Context, VectorIndex
from autointent.configs import EmbedderConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer


class MLKnnScorer(BaseScorer):
    """Multi-label k-nearest neighbors (ML-KNN) scorer.

    This module implements ML-KNN, a multi-label classifier that computes probabilities
    based on the k-nearest neighbors of a query instance.

    Args:
        k: Number of nearest neighbors to consider
        embedder_config: Config of the embedder used for vectorization
        s: Smoothing parameter for probability calculations, defaults to 1.0
        ignore_first_neighbours: Number of closest neighbors to ignore, defaults to 0

    Example:
    --------

    .. testcode::

        from autointent.modules.scoring import MLKnnScorer
        utterances = ["what is your name?", "how are you?"]
        labels = [[1,0], [0,1]]
        scorer = MLKnnScorer(
            k=5,
            embedder_config="sergeyzh/rubert-tiny-turbo",
        )
        scorer.fit(utterances, labels)
        test_utterances = ["Hi!", "What's up?"]
        probabilities = scorer.predict(test_utterances)
        print(probabilities)  # Outputs predicted probabilities for each label

    .. testoutput::

        [[0.5 0.5]
         [0.5 0.5]]

    """

    name = "mlknn"
    _n_classes: int
    _vector_index: VectorIndex
    _prior_prob_true: NDArray[Any]
    _prior_prob_false: NDArray[Any]
    _cond_prob_true: NDArray[Any]
    _cond_prob_false: NDArray[Any]
    _features: NDArray[Any]
    _labels: NDArray[Any]
    supports_multiclass = False
    supports_multilabel = True

    def __init__(
        self,
        k: PositiveInt = 5,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        s: float = 1.0,
        ignore_first_neighbours: int = 0,
    ) -> None:
        self.k = k
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self.s = s
        self.ignore_first_neighbours = ignore_first_neighbours

        if self.k < 0 or not isinstance(self.k, int):
            msg = "`k` argument of `MLKnnScorer` must be a positive int"
            raise ValueError(msg)

        if not isinstance(self.s, float | int):
            assert_never(self.s)

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: PositiveInt = 5,
        s: PositiveFloat = 1.0,
        ignore_first_neighbours: NonNegativeInt = 0,
        embedder_config: EmbedderConfig | str | None = None,
    ) -> "MLKnnScorer":
        """Create an MLKnnScorer instance using a Context object.

        Args:
            context: Context containing configurations and utilities
            k: Number of nearest neighbors to consider
            s: Smoothing parameter for probability calculations, defaults to 1.0
            ignore_first_neighbours: Number of closest neighbors to ignore, defaults to 0
            embedder_config: Config of the embedder, or None to use the best embedder

        Returns:
            Initialized MLKnnScorer instance
        """
        if embedder_config is None:
            embedder_config = context.resolve_embedder()

        return cls(
            k=k,
            embedder_config=embedder_config,
            s=s,
            ignore_first_neighbours=ignore_first_neighbours,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {"embedder_config": self.embedder_config.model_dump()}

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        """Fit the scorer by training or loading the vector index and calculating probabilities.

        Args:
            utterances: List of training utterances
            labels: List of multi-label targets for each utterance

        Raises:
            TypeError: If the labels are not multi-label
            ValueError: If the vector index mismatches the provided utterances
        """
        self._validate_task(labels)

        self._vector_index = VectorIndex(
            EmbedderConfig(
                model_name=self.embedder_config.model_name,
                device=self.embedder_config.device,
                batch_size=self.embedder_config.batch_size,
                tokenizer_config=self.embedder_config.tokenizer_config,
                use_cache=self.embedder_config.use_cache,
            ),
        )
        self._vector_index.add(utterances, labels)

        self._features = self._vector_index.get_all_embeddings()
        self._labels = np.array(labels)
        self._prior_prob_true, self._prior_prob_false = self._compute_prior(self._labels)
        self._cond_prob_true, self._cond_prob_false = self._compute_cond()

    def _compute_prior(self, y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute prior probabilities for each class.

        Args:
            y: Array of labels (multi-label format)

        Returns:
            Tuple of prior probabilities for true and false labels
        """
        prior_prob_true = (self.s + y.sum(axis=0)) / (self.s * 2 + y.shape[0])
        prior_prob_false = 1 - prior_prob_true
        return prior_prob_true, prior_prob_false

    def _compute_cond(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute conditional probabilities for neighbors.

        Returns:
            Tuple of conditional probabilities for true and false labels
        """
        c = np.zeros((self._n_classes, self.k + 1), dtype=int)
        cn = np.zeros((self._n_classes, self.k + 1), dtype=int)

        neighbors_labels, _ = self._get_neighbors(self._features)

        for i in range(self._labels.shape[0]):
            deltas = np.sum(neighbors_labels[i], axis=0).astype(int)
            idx_helper = np.arange(self._n_classes)
            deltas_idx = deltas[idx_helper]
            c[idx_helper, deltas_idx] += self._labels[i]
            cn[idx_helper, deltas_idx] += 1 - self._labels[i]

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_prob_true = (self.s + c) / (self.s * (self.k + 1) + c_sum[:, None])
        cond_prob_false = (self.s + cn) / (self.s * (self.k + 1) + cn_sum[:, None])

        return cond_prob_true, cond_prob_false

    def _get_neighbors(
        self,
        queries: list[str] | NDArray[Any],
    ) -> tuple[NDArray[np.int64], list[list[str]]]:
        """Get nearest neighbors for given queries.

        Args:
            queries: List of query utterances or embedded features

        Returns:
            Tuple containing:
                - Array of neighbor labels
                - List of neighbor utterances
        """
        labels, _, neighbors = self._vector_index.query(
            queries,
            self.k + self.ignore_first_neighbours,
        )
        return (
            np.array([candidates[self.ignore_first_neighbours :] for candidates in labels]),
            neighbors,
        )

    def predict_labels(self, utterances: list[str], thresh: float = 0.5) -> NDArray[np.int64]:
        """Predict labels for the given utterances.

        Args:
            utterances: List of query utterances
            thresh: Threshold for binary classification, defaults to 0.5

        Returns:
            Predicted labels as a binary array
        """
        probas = self.predict(utterances)
        return (probas > thresh).astype(int)

    def predict(self, utterances: list[str]) -> NDArray[np.float64]:
        """Predict probabilities for the given utterances.

        Args:
            utterances: List of query utterances

        Returns:
            Array of predicted probabilities for each class
        """
        return self._predict(utterances)[0]

    def predict_with_metadata(self, utterances: list[str]) -> tuple[NDArray[Any], list[dict[str, Any]] | None]:
        """Predict probabilities along with metadata for the given utterances.

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
            delattr(self, "_prior_prob_true")
            delattr(self, "_prior_prob_false")
            delattr(self, "_cond_prob_true")
            delattr(self, "_cond_prob_false")
            delattr(self, "_features")
            delattr(self, "_labels")

    def _predict(
        self,
        utterances: list[str],
    ) -> tuple[NDArray[np.float64], list[list[str]]]:
        """Predict probabilities and retrieve neighbors for the given utterances.

        Args:
            utterances: List of query utterances

        Returns:
            Tuple containing:
                - Array of predicted probabilities
                - List of neighbor utterances
        """
        result = np.zeros((len(utterances), self._n_classes), dtype=float)
        neighbors_labels, neighbors = self._get_neighbors(utterances)

        for instance in range(neighbors_labels.shape[0]):
            deltas = np.sum(neighbors_labels[instance], axis=0).astype(int)

            for label in range(self._n_classes):
                p_true = self._prior_prob_true[label] * self._cond_prob_true[label, deltas[label]]
                p_false = self._prior_prob_false[label] * self._cond_prob_false[label, deltas[label]]
                result[instance, label] = p_true / (p_true + p_false)

        return result, neighbors
