"""RerankScorer class for re-ranking based on cross-encoder scoring."""

from typing import Any

import numpy as np
import numpy.typing as npt

from autointent import Context, Ranker
from autointent.custom_types import WEIGHT_TYPES, LabelType

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

    def __init__(  # noqa: PLR0913
        self,
        cross_encoder_name: str,
        embedder_name: str,
        k: int,
        weights: WEIGHT_TYPES,
        m: int | None = None,
        rank_threshold_cutoff: int | None = None,
        embedder_device: str = "cpu",
        embedder_batch_size: int = 32,
        embedder_max_length: int | None = None,
        cross_encoder_device: str = "cpu",
        cross_encoder_batch_size: int = 32,
        cross_encoder_max_length: int | None = None,
        embedder_use_cache: bool = True,
        train_head: bool = False,
    ) -> None:
        """
        Initialize the RerankScorer.

        :param embedder_name: Name of the embedder used for vectorization.
        :param k: Number of closest neighbors to consider during inference.
        :param weights: Weighting strategy:
            - "uniform" (or False): Equal weight for all neighbors.
            - "distance" (or True): Weight inversely proportional to distance.
            - "closest": Only the closest neighbor of each class is weighted.
        :param cross_encoder_name: Name of the cross-encoder model used for re-ranking.
        :param m: Number of top-ranked neighbors to consider, or None to use k.
        :param rank_threshold_cutoff: Rank threshold cutoff for re-ranking, or None.
        :param embedder_device: Device to run operations on, e.g., "cpu" or "cuda".
        :param embedder_batch_size: Batch size for embedding generation, defaults to 32.
        :param embedder_max_length: Maximum sequence length for embedding and cross encoder, or None for default.
        """
        super().__init__(
            embedder_name=embedder_name,
            k=k,
            weights=weights,
            embedder_device=embedder_device,
            embedder_batch_size=embedder_batch_size,
            embedder_max_length=embedder_max_length,
            embedder_use_cache=embedder_use_cache,
        )

        self.cross_encoder_name = cross_encoder_name
        self.cross_encoder_device = cross_encoder_device
        self.cross_encoder_batch_size = cross_encoder_batch_size
        self.cross_encoder_max_length = cross_encoder_max_length
        self.train_head = train_head

        self.m = k if m is None else m
        self.rank_threshold_cutoff = rank_threshold_cutoff

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: int,
        weights: WEIGHT_TYPES,
        cross_encoder_name: str,
        train_head: bool = False,
        embedder_name: str | None = None,
        m: int | None = None,
        rank_threshold_cutoff: int | None = None,
    ) -> "RerankScorer":
        """
        Create a RerankScorer instance from a given context.

        :param context: Context object containing optimization information and vector index client.
        :param k: Number of closest neighbors to consider during inference.
        :param weights: Weighting strategy.
        :param cross_encoder_name: Name of the cross-encoder model used for re-ranking.
        :param embedder_name: Name of the embedder used for vectorization, or None to use the best existing embedder.
        :param m: Number of top-ranked neighbors to consider, or None to use k.
        :param rank_threshold_cutoff: Rank threshold cutoff for re-ranking, or None.
        :return: An instance of RerankScorer.
        """
        if embedder_name is None:
            embedder_name = context.optimization_info.get_best_embedder()

        return cls(
            k=k,
            weights=weights,
            m=m,
            rank_threshold_cutoff=rank_threshold_cutoff,
            train_head=train_head,
            embedder_name=embedder_name,
            embedder_device=context.get_device(),
            embedder_batch_size=context.get_batch_size(),
            embedder_max_length=context.get_max_length(),
            embedder_use_cache=context.get_use_cache(),
            cross_encoder_name=cross_encoder_name,
            cross_encoder_device=context.get_cross_encoder_device(),
            cross_encoder_batch_size=context.get_cross_encoder_batch_size(),
            cross_encoder_max_length=context.get_cross_encoder_max_length(),
        )

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        """
        Fit the RerankScorer with utterances and labels.

        :param utterances: List of utterances to fit the scorer.
        :param labels: List of labels corresponding to the utterances.
        """
        self._scorer = Ranker(
            self.cross_encoder_name,
            device=self.cross_encoder_device,
            max_length=self.cross_encoder_max_length,
            batch_size=self.cross_encoder_batch_size,
            train_classifier=self.train_head,
        )
        self._scorer.fit(utterances, labels)

        super().fit(utterances, labels)

    def _predict(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[list[str]]]:
        """
        Predict the scores and neighbors for given utterances.

        :param utterances: List of utterances to predict scores for.
        :return: A tuple containing the scores and neighbors.
        """
        knn_labels, knn_distances, knn_neighbors = self._get_neighbours(utterances)

        labels: list[list[LabelType]] = []
        distances: list[list[float]] = []
        neighbours: list[list[str]] = []

        for query, query_labels, query_distances, query_docs in zip(
            utterances, knn_labels, knn_distances, knn_neighbors, strict=True
        ):
            cur_ranks = self._scorer.rank(query, query_docs, top_k=self.m)

            for dst, src in zip(
                [labels, distances, neighbours], [query_labels, query_distances, query_docs], strict=True
            ):
                dst.append([src[rank["corpus_id"]] for rank in cur_ranks])  # type: ignore[attr-defined, index]

        scores = self._count_scores(np.array(labels), np.array(distances))
        return scores, neighbours
