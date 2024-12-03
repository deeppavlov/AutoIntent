import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sentence_transformers import CrossEncoder
from torch.nn import Sigmoid
from typing_extensions import Self

from autointent.context import Context
from autointent.custom_types import WEIGHT_TYPES, LabelType

from .knn import KNNScorer, KNNScorerDumpMetadata


class RerankScorerDumpMetadata(KNNScorerDumpMetadata):
    scorer_name: str
    m: int | None
    rank_threshold_cutoff: int | None


class RerankScorer(KNNScorer):
    name = "rerank_scorer"
    _scorer: CrossEncoder

    def __init__(
        self,
        embedder_name: str,
        k: int,
        weights: WEIGHT_TYPES,
        scorer_name: str,
        m: int | None = None,
        rank_threshold_cutoff: int | None = None,
        db_dir: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int | None = None,
    ) -> None:
        super().__init__(
            embedder_name=embedder_name,
            k=k,
            weights=weights,
            db_dir=db_dir,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

        self.scorer_name = scorer_name
        self.m = k if m is None else m
        self.rank_threshold_cutoff = rank_threshold_cutoff
        self._scorer = CrossEncoder(self.scorer_name, device=self.device, max_length=self.max_length)  # type: ignore[arg-type]

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: int,
        weights: WEIGHT_TYPES,
        scorer_name: str,
        embedder_name: str | None = None,
        m: int | None = None,
        rank_threshold_cutoff: int | None = None,
    ) -> Self:
        if embedder_name is None:
            embedder_name = context.optimization_info.get_best_embedder()
            prebuilt_index = True
        else:
            prebuilt_index = context.vector_index_client.exists(embedder_name)

        instance = cls(
            embedder_name=embedder_name,
            k=k,
            weights=weights,
            scorer_name=scorer_name,
            m=m,
            rank_threshold_cutoff=rank_threshold_cutoff,
            db_dir=str(context.get_db_dir()),
            device=context.get_device(),
            batch_size=context.get_batch_size(),
            max_length=context.get_max_length(),
        )
        # TODO: needs re-thinking....
        instance.prebuilt_index = prebuilt_index
        return instance

    def _store_state_to_metadata(self) -> RerankScorerDumpMetadata:
        return RerankScorerDumpMetadata(
            **super()._store_state_to_metadata(),
            m=self.m,
            scorer_name=self.scorer_name,
            rank_threshold_cutoff=self.rank_threshold_cutoff,
        )

    def load(self, path: str) -> None:
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            self.metadata: RerankScorerDumpMetadata = json.load(file)

        self._restore_state_from_metadata(self.metadata)

    def _restore_state_from_metadata(self, metadata: RerankScorerDumpMetadata) -> None:
        super()._restore_state_from_metadata(metadata)

        self.m = metadata["m"] if metadata["m"] else self.k
        self.scorer_name = metadata["scorer_name"]
        self.rank_threshold_cutoff = metadata["rank_threshold_cutoff"]

    def _predict(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[list[str]]]:
        knn_labels, knn_distances, knn_neighbors = self._get_neighbours(utterances)

        labels: list[list[LabelType]] = []
        distances: list[list[float]] = []
        neighbours: list[list[str]] = []

        for query, query_labels, query_distances, query_docs in zip(
            utterances, knn_labels, knn_distances, knn_neighbors, strict=True
        ):
            cur_ranks = self._scorer.rank(
                query, query_docs, top_k=self.m, batch_size=self.batch_size, activation_fct=Sigmoid()
            )
            # if self.rank_threshold_cutoff:
            #     # remove neighbours where CrossEncoder is not confident enough
            #     while len(cur_ranks):
            #         if cur_ranks[-1]['score'] >= self.rank_threshold_cutoff:
            #             break
            #         cur_ranks.pop()

            # keep only relevant data for the utterance
            for dst, src in zip([labels, distances, neighbours], [query_labels, query_distances, query_docs],
                                strict=True):
                dst.append([src[rank["corpus_id"]] for rank in cur_ranks])  # type: ignore[attr-defined, index]

        scores = self._count_scores(np.array(labels), np.array(distances))
        return scores, neighbours
