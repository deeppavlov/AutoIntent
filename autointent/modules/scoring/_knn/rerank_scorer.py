"""RerankScorer class for re-ranking based on cross-encoder scoring."""

from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context, Ranker
from autointent.custom_types import WEIGHT_TYPES, ListOfLabels
from autointent.schemas import CrossEncoderConfig, EmbedderConfig

from .knn import KNNScorer


class RerankScorer(KNNScorer):
    """
    Re-ranking scorer using a cross-encoder for intent classification.

    This module uses a cross-encoder to re-rank the nearest neighbors retrieved by a KNN scorer.

    :ivar name: Name of the scorer, defaults to "rerank".
    :ivar _scorer: Ranker instance for re-ranking.
    """

    name = "rerank"
    _scorer: Ranker

    def __init__(
        self,
        cross_encoder_config: CrossEncoderConfig | str | dict[str, Any],
        embedder_config: EmbedderConfig | str | dict[str, Any],
        k: int,
        weights: WEIGHT_TYPES,
        m: int | None = None,
        rank_threshold_cutoff: int | None = None,
    ) -> None:
        """
        Initialize the RerankScorer.

        :param embedder_config: Config of the embedder used for vectorization.
        :param k: Number of closest neighbors to consider during inference.
        :param weights: Weighting strategy:
            - "uniform": Equal weight for all neighbors.
            - "distance": Weight inversely proportional to distance.
            - "closest": Only the closest neighbor of each class is weighted.
        :param cross_encoder_config: Config of the cross-encoder model used for re-ranking.
        :param m: Number of top-ranked neighbors to consider, or None to use k.
        :param rank_threshold_cutoff: Rank threshold cutoff for re-ranking, or None.
        """
        super().__init__(
            embedder_config=embedder_config,
            k=k,
            weights=weights,
        )

        self.cross_encoder_config = CrossEncoderConfig.from_search_config(cross_encoder_config)

        self.m = k if m is None else m
        self.rank_threshold_cutoff = rank_threshold_cutoff

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: int,
        weights: WEIGHT_TYPES,
        cross_encoder_config: CrossEncoderConfig | str,
        embedder_config: EmbedderConfig | str | None = None,
        m: int | None = None,
        rank_threshold_cutoff: int | None = None,
    ) -> "RerankScorer":
        """
        Create a RerankScorer instance from a given context.

        :param context: Context object containing optimization information and vector index client.
        :param k: Number of closest neighbors to consider during inference.
        :param weights: Weighting strategy.
        :param cross_encoder_config: Config of the cross-encoder model used for re-ranking.
        :param embedder_config: Config of the embedder used for vectorization,
            or None to use the best existing embedder.
        :param m: Number of top-ranked neighbors to consider, or None to use k.
        :param rank_threshold_cutoff: Rank threshold cutoff for re-ranking, or None.
        :return: An instance of RerankScorer.
        """
        if embedder_config is None:
            embedder_config = context.optimization_info.get_best_embedder()

        return cls(
            k=k,
            weights=weights,
            m=m,
            rank_threshold_cutoff=rank_threshold_cutoff,
            embedder_config=embedder_config,
            cross_encoder_config=cross_encoder_config,
        )

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        """
        Fit the RerankScorer with utterances and labels.

        :param utterances: List of utterances to fit the scorer.
        :param labels: List of labels corresponding to the utterances.
        """
        if hasattr(self, "_scorer"):
            self.clear_cache()

        self._scorer = Ranker(
            self.cross_encoder_config,
        )
        self._scorer.fit(utterances, labels)

        super().fit(utterances, labels, clear_cache=False)

    def clear_cache(self) -> None:
        self._scorer.clear_ram()
        super().clear_cache()

    def _predict(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[list[str]]]:
        """
        Predict the scores and neighbors for given utterances.

        :param utterances: List of utterances to predict scores for.
        :return: A tuple containing the scores and neighbors.
        """
        knn_labels, knn_distances, knn_neighbors = self._get_neighbours(utterances)

        labels: list[ListOfLabels] = []
        distances: list[list[float]] = []
        neighbours: list[list[str]] = []

        for query, query_labels, query_distances, query_docs in zip(
            utterances, knn_labels, knn_distances, knn_neighbors, strict=True
        ):
            cur_ranks = self._scorer.rank(query, query_docs, top_k=self.m)

            for dst, src in zip(
                [labels, distances, neighbours], [query_labels, query_distances, query_docs], strict=True
            ):
                dst.append([src[rank["corpus_id"]] for rank in cur_ranks])  # type: ignore[attr-defined]

        scores = self._count_scores(np.array(labels), np.array(distances))
        return scores, neighbours
