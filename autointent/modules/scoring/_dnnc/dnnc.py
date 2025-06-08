"""DNNCScorer class for scoring utterances using deep neural network classifiers (DNNC)."""

import itertools as it
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import PositiveInt

from autointent import Context, Ranker, VectorIndex
from autointent.configs import CrossEncoderConfig, EmbedderConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer

logger = logging.getLogger(__name__)


class DNNCScorer(BaseScorer):
    """Scoring module for intent classification using discriminative nearest neighbor classification.

    This module uses a Ranker for scoring candidate intents and can optionally
    train a logistic regression head on top of cross-encoder features.

    Args:
            cross_encoder_config: Config of the cross-encoder model
            embedder_config: Config of the embedder model
            k: Number of nearest neighbors to retrieve

    Examples:
    --------

    .. testcode::

        from autointent.modules.scoring import DNNCScorer
        utterances = ["what is your name?", "how are you?"]
        labels = [0, 1]
        scorer = DNNCScorer(
            cross_encoder_config="cross-encoder/ms-marco-MiniLM-L6-v2",
            embedder_config="sergeyzh/rubert-tiny-turbo",
            k=5,
        )
        scorer.fit(utterances, labels)

        test_utterances = ["Hello!", "What's up?"]
        scores = scorer.predict(test_utterances)

    Reference:
        Zhang, J. G., Hashimoto, K., Liu, W., Wu, C. S., Wan, Y., Yu, P. S., ... & Xiong, C. (2020).
        Discriminative Nearest Neighbor Few-Shot Intent Detection by Transferring Natural Language Inference.
        arXiv preprint arXiv:2010.13009.

    """

    name = "dnnc"
    _n_classes: int
    _vector_index: VectorIndex
    _cross_encoder: Ranker
    supports_multilabel = False
    supports_multiclass = True

    def __init__(
        self,
        k: PositiveInt = 5,
        cross_encoder_config: CrossEncoderConfig | str | dict[str, Any] | None = None,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
    ) -> None:
        self.cross_encoder_config = CrossEncoderConfig.from_search_config(cross_encoder_config)
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self.k = k

        if self.k < 0 or not isinstance(self.k, int):
            msg = "`k` argument of `DNNCScorer` must be a positive int"
            raise ValueError(msg)

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: PositiveInt = 5,
        cross_encoder_config: CrossEncoderConfig | str | None = None,
        embedder_config: EmbedderConfig | str | None = None,
    ) -> "DNNCScorer":
        """Create a DNNCScorer instance using a Context object.

        Args:
            context: Context containing configurations and utilities
            cross_encoder_config: Config of the cross-encoder model
            k: Number of nearest neighbors to retrieve
            embedder_config: Config of the embedder model, or None to use the best embedder
        """
        if embedder_config is None:
            embedder_config = context.resolve_embedder()

        if cross_encoder_config is None:
            cross_encoder_config = context.resolve_ranker()

        return cls(
            k=k,
            embedder_config=embedder_config,
            cross_encoder_config=cross_encoder_config,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {
            "embedder_config": self.embedder_config.model_dump(),
            "cross_encoder_config": self.cross_encoder_config.model_dump(),
        }

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        """Fit the scorer by training or loading the vector index.

        Args:
            utterances: List of training utterances
            labels: List of labels corresponding to the utterances

        Raises:
            ValueError: If the vector index mismatches the provided utterances
        """
        self._validate_task(labels)

        self._vector_index = VectorIndex(self.embedder_config)
        self._vector_index.add(utterances, labels)

        self._cross_encoder = Ranker(self.cross_encoder_config, output_range="sigmoid")
        self._cross_encoder.fit(utterances, labels)

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """Predict class scores for the given utterances.

        Args:
            utterances: List of utterances to score

        Returns:
            Array of predicted scores
        """
        return self._predict(utterances)[0]

    def predict_with_metadata(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[dict[str, Any]] | None]:
        """Predict class scores along with metadata for the given utterances.

        Args:
            utterances: List of utterances to score

        Returns:
            Tuple containing:
                - Array of predicted scores
                - List of metadata with neighbor details and scores
        """
        scores, neighbors, neighbors_scores = self._predict(utterances)
        metadata = [
            {"neighbors": utterance_neighbors, "scores": utterance_neighbors_scores}
            for utterance_neighbors, utterance_neighbors_scores in zip(neighbors, neighbors_scores, strict=True)
        ]
        return scores, metadata

    def _get_cross_encoder_scores(self, utterances: list[str], candidates: list[list[str]]) -> list[list[float]]:
        """Compute cross-encoder scores for utterances against their candidate neighbors.

        Args:
            utterances: List of query utterances
            candidates: List of candidate utterances for each query

        Returns:
            List of cross-encoder scores for each query-candidate pair

        Raises:
            ValueError: If the number of utterances and candidates do not match
        """
        if len(utterances) != len(candidates):
            msg = "Number of utterances doesn't match number of retrieved candidates"
            logger.error(msg)
            raise ValueError(msg)

        text_pairs = [[(query, cand) for cand in docs] for query, docs in zip(utterances, candidates, strict=False)]

        flattened_text_pairs = list(it.chain.from_iterable(text_pairs))

        if len(flattened_text_pairs) != len(utterances) * len(candidates[0]):
            msg = "Number of candidates for each query utterance cannot vary"
            logger.error(msg)
            raise ValueError(msg)

        flattened_cross_encoder_scores: npt.NDArray[np.float64] = self._cross_encoder.predict(flattened_text_pairs)
        return [
            flattened_cross_encoder_scores[i : i + self.k].tolist()
            for i in range(0, len(flattened_cross_encoder_scores), self.k)
        ]

    def _build_result(self, scores: list[list[float]], labels: list[ListOfLabels]) -> npt.NDArray[Any]:
        """Build a result matrix with scores assigned to the best neighbor's class.

        Args:
            scores: Cross encoder scores for each query's k closest utterances
            labels: Corresponding intent labels

        Returns:
            Matrix of shape (n_queries, n_classes) with zeros everywhere except the class of the best neighbor
        """
        return build_result(np.array(scores), np.array(labels), self._n_classes)

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the vector index."""
        if hasattr(self, "_vector_index"):
            self._vector_index.clear_ram()
            self._cross_encoder.clear_ram()

    def _predict(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[list[str]], list[list[float]]]:
        """Predict class scores using vector index and cross-encoder.

        Args:
            utterances: List of query utterances

        Returns:
            Tuple containing:
                - Class scores matrix
                - List of neighbor utterances
                - List of neighbor scores
        """
        labels, _, neighbors = self._vector_index.query(
            utterances,
            self.k,
        )

        cross_encoder_scores = self._get_cross_encoder_scores(utterances, neighbors)

        return self._build_result(cross_encoder_scores, labels), neighbors, cross_encoder_scores


def build_result(scores: npt.NDArray[Any], labels: npt.NDArray[Any], n_classes: int) -> npt.NDArray[Any]:
    """Build a result matrix with scores assigned to the best neighbor's class.

    Args:
        scores: Cross-encoder scores for each query's neighbors
        labels: Labels corresponding to each neighbor
        n_classes: Total number of classes

    Returns:
        Matrix of shape (n_queries, n_classes) with scores for the best class
    """
    res = np.zeros((len(scores), n_classes))
    best_neighbors = np.argmax(scores, axis=1)
    idx_helper = np.arange(len(res))
    best_classes = labels[idx_helper, best_neighbors]
    best_scores = scores[idx_helper, best_neighbors]
    res[idx_helper, best_classes] = best_scores
    return res
