import itertools as it
import logging

import numpy as np
from typing import Any
import numpy.typing as npt
from sentence_transformers import CrossEncoder

from autointent.modules.scoring.base import Context, ScoringModule

from .head_training import CrossEncoderWithLogreg

logger = logging.getLogger(__name__)


class DNNCScorer(ScoringModule):
    """
    TODO:
    - think about other cross-encoder settings
    - implement training of cross-encoder with sentence_encoders utils
    - inspect batch size of model.predict?
    """

    def __init__(self, model_name: str, k: int, train_head: bool = False) -> None:
        self.model_name = model_name
        self.k = k
        self.train_head = train_head

    def fit(self, context: Context) -> None:
        self.model = CrossEncoder(self.model_name, trust_remote_code=True, device=context.device)
        self._collection = context.get_best_collection()

        if self.train_head:
            model = CrossEncoderWithLogreg(self.model)
            model.fit(context.data_handler.utterances_train, context.data_handler.labels_train)
            self.model = model

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """
        Return
        ---
        `(n_queries, n_classes)` matrix with zeros everywhere except the class of the best neighbor utterance
        """
        query_res = self._collection.query(
            query_texts=utterances,
            n_results=self.k,
            include=["metadatas", "documents"],  # one can add "embeddings", "distances"
        )

        cross_encoder_scores = self._get_cross_encoder_scores(utterances, query_res["documents"])

        labels_pred = [[cand["intent_id"] for cand in candidates] for candidates in query_res["metadatas"]]

        return self._build_result(cross_encoder_scores, labels_pred)

    def _get_cross_encoder_scores(
        self, utterances: list[str], candidates: list[list[str]]
    ) -> list[list[float]]:
        """
        Arguments
        ---
        `utterances`: list of query utterances
        `candidates`: for each query, this list contains a list of the k the closest sample utterances \
            (from retrieval module)

        Return
        ---
        for each query, return a list of a corresponding cross encoder scores for the k the closest sample utterances
        """
        if len(utterances) != len(candidates):
            msg = "Number of utterances doesn't match number of retrieved candidates"
            logger.error(msg)
            raise ValueError(msg)

        text_pairs = [[[query, cand] for cand in docs] for query, docs in zip(utterances, candidates, strict=False)]

        flattened_text_pairs = list(it.chain.from_iterable(text_pairs))

        if len(flattened_text_pairs) != len(utterances) * len(candidates[0]):
            msg = "Number of candidates for each query utterance cannot vary"
            logger.error(msg)
            raise ValueError(msg)

        flattened_cross_encoder_scores = self.model.predict(flattened_text_pairs)
        return [
            flattened_cross_encoder_scores[i : i + self.k]
            for i in range(0, len(flattened_cross_encoder_scores), self.k)
        ]

    def _build_result(
        self, scores: list[list[float]], labels: list[list[int]]
    ) -> npt.NDArray[Any]:
        """
        Arguments
        ---
        `scores`: for each query utterance, cross encoder scores of its k closest utterances
        `labels`: corresponding intent labels

        Return
        ---
        `(n_queries, n_classes)` matrix with zeros everywhere except the class of the best neighbor utterance
        """
        n_classes = self._collection.metadata["n_classes"]

        return build_result(np.array(scores), np.array(labels), n_classes)

    def clear_cache(self) -> None:
        model = self._collection._embedding_function._model  # noqa: SLF001
        model.to(device="cpu")
        del model
        self._collection = None


def build_result(
    scores: npt.NDArray[Any], labels: npt.NDArray[Any], n_classes: int
) -> npt.NDArray[Any]:
    res = np.zeros((len(scores), n_classes))
    best_neighbors = np.argmax(scores, axis=1)
    idx_helper = np.arange(len(res))
    best_classes = labels[idx_helper, best_neighbors]
    best_scores = scores[idx_helper, best_neighbors]
    res[idx_helper, best_classes] = best_scores
    return res
