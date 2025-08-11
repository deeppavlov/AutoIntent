"""RerankScorer class for re-ranking based on cross-encoder scoring."""

from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import PositiveInt

from autointent import Context, Ranker
from autointent.configs import CrossEncoderConfig, EmbedderConfig
from autointent.custom_types import ListOfLabels, WeightType

from .knn import KNNScorer


class RerankScorer(KNNScorer):
    """Re-ranking scorer using a cross-encoder for intent classification.

    This module uses a cross-encoder to re-rank the nearest neighbors retrieved by a KNN scorer.

    Args:
        embedder_config: Config of the embedder used for vectorization
        k: Number of closest neighbors to consider during inference
        weights: Weighting strategy:

            - "uniform": Equal weight for all neighbors
            - "distance": Weight inversely proportional to distance
            - "closest": Only the closest neighbor of each class is weighted

        cross_encoder_config: Config of the cross-encoder model used for re-ranking
        m: Number of top-ranked neighbors to consider, or None to use k
    """

    name = "rerank"
    _scorer: Ranker

    def __init__(
        self,
        k: PositiveInt = 5,
        weights: WeightType = "distance",
        use_cross_encoder_scores: bool = False,
        m: PositiveInt | None = None,
        cross_encoder_config: CrossEncoderConfig | str | dict[str, Any] | None = None,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            embedder_config=embedder_config,
            k=k,
            weights=weights,
        )

        self.cross_encoder_config = CrossEncoderConfig.from_search_config(cross_encoder_config)

        self.m = k if m is None else m
        self.use_cross_encoder_scores = use_cross_encoder_scores

        if self.m < 0 or not isinstance(self.m, int):
            msg = "`m` argument of `RerankScorer` must be a positive int"
            raise ValueError(msg)

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: PositiveInt = 5,
        weights: WeightType = "distance",
        m: PositiveInt | None = None,
        cross_encoder_config: CrossEncoderConfig | str | None = None,
        embedder_config: EmbedderConfig | str | None = None,
        use_cross_encoder_scores: bool = False,
    ) -> "RerankScorer":
        """Create a RerankScorer instance from a given context.

        Args:
            context: Context object containing optimization information and vector index client
            k: Number of closest neighbors to consider during inference
            weights: Weighting strategy
            cross_encoder_config: Config of the cross-encoder model used for re-ranking
            embedder_config: Config of the embedder used for vectorization,
                or None to use the best existing embedder
            m: Number of top-ranked neighbors to consider, or None to use k
            use_cross_encoder_scores: use crosencoder scores for the output probability vector computation
        """
        if embedder_config is None:
            embedder_config = context.resolve_embedder()

        if cross_encoder_config is None:
            cross_encoder_config = context.resolve_ranker()

        return cls(
            k=k,
            weights=weights,
            m=m,
            use_cross_encoder_scores=use_cross_encoder_scores,
            embedder_config=embedder_config,
            cross_encoder_config=cross_encoder_config,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {
            "embedder_config": self.embedder_config.model_dump(),
            "cross_encoder_config": self.cross_encoder_config.model_dump(),
        }

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        """Fit the RerankScorer with utterances and labels.

        Args:
            utterances: List of utterances to fit the scorer
            labels: List of labels corresponding to the utterances
        """
        self._scorer = Ranker(self.cross_encoder_config, output_range="tanh")
        self._scorer.fit(utterances, labels)

        super().fit(utterances, labels)

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the scorer and vector index."""
        if hasattr(self, "_scorer"):
            self._scorer.clear_ram()
            super().clear_cache()

    def _predict(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[list[str]]]:
        """Predict the scores and neighbors for given utterances.

        Args:
            utterances: List of utterances to predict scores for

        Returns:
            Tuple containing:
                - Array of predicted scores
                - List of neighbor utterances
        """
        knn_labels, knn_distances, knn_neighbors = self._get_neighbours(utterances)

        labels: list[ListOfLabels] = []
        distances: list[list[float]] = []
        neighbours: list[list[str]] = []

        for query, query_labels, query_distances, query_docs in zip(
            utterances, knn_labels, knn_distances, knn_neighbors, strict=True
        ):
            cur_ranks = self._scorer.rank(query, query_docs, top_k=self.m)

            for dst, src in zip([labels, neighbours], [query_labels, query_docs], strict=True):
                dst.append([src[rank["corpus_id"]] for rank in cur_ranks])  # type: ignore[attr-defined]

            if self.use_cross_encoder_scores:
                distances.append([rank["score"] for rank in cur_ranks])
            else:
                distances.append([query_distances[rank["corpus_id"]] for rank in cur_ranks])

        scores = self._count_scores(np.array(labels), np.array(distances))

        return scores, neighbours
