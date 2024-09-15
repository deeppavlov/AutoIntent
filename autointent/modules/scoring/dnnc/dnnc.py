import itertools as it

import numpy as np
from sentence_transformers import CrossEncoder
from ..base import Context, ScoringModule
from .head_training import CrossEncoderWithLogreg

import logging

logger = logging.getLogger(__name__)
class DNNCScorer(ScoringModule):
    """
    TODO:
    - think about other cross-encoder settings
    - implement training of cross-encoder with sentence_encoders utils
    - inspect batch size of model.predict?
    """

    def __init__(self, model_name: str, k: int, train_head=False):
        self.model_name = model_name
        self.k = k
        self.train_head = train_head

    def fit(self, context: Context):
        self.model = CrossEncoder(self.model_name, trust_remote_code=True, device=context.device)
        self._collection = context.get_best_collection()

        if self.train_head:
            model = CrossEncoderWithLogreg(self.model)
            model.fit(context.data_handler.utterances_train, context.data_handler.labels_train)
            self.model = model

    def predict(self, utterances: list[str]):
        """
        Return
        ---
        `(n_queries, n_classes)` matrix with zeros everywhere except the class of the best neighbor utterance
        """
        logger.info(f"Starting prediction for {len(utterances)} utterances")

        query_res = self._collection.query(
            query_texts=utterances,
            n_results=self.k,
            include=["metadatas", "documents"],  # one can add "embeddings", "distances"
        )

        logger.info(f"Query results obtained. Metadata count: {len(query_res['metadatas'])}")
        logger.info(
            f"First metadata item: {query_res['metadatas'][0] if query_res['metadatas'] else 'Empty'}")

        cross_encoder_scores = self._get_cross_encoder_scores(
            utterances, query_res["documents"]
        )

        logger.info(f"Cross encoder scores calculated. Type: {type(cross_encoder_scores)}")
        logger.info(f"Cross encoder scores length: {len(cross_encoder_scores)}")
        if cross_encoder_scores:
            logger.info(f"First cross encoder score: {cross_encoder_scores[0]}")
            if isinstance(cross_encoder_scores[0], list):
                logger.info(f"Length of first cross encoder score: {len(cross_encoder_scores[0])}")

        logger.info("Processing metadata to extract intent_ids")
        labels_pred = []
        for i, candidates in enumerate(query_res["metadatas"]):
            try:
                intent_ids = [int(next((k for k, v in cand.items() if v == 1), -1)) for cand in candidates]
                logger.info(f"Intent IDs for utterance {i}: {intent_ids}")
                labels_pred.append(intent_ids)
            except Exception as e:
                logger.error(f"Error encountered in metadata {i}: {e}")
                logger.error(f"Problematic metadata: {candidates}")
                raise

        logger.info(f"Labels predicted. Count: {len(labels_pred)}")

        res = self._build_result(cross_encoder_scores, labels_pred)

        logger.info(f"Final result built. Type: {type(res)}")
        if isinstance(res, np.ndarray):
            logger.info(f"Final result shape: {res.shape}")
        else:
            logger.info(f"Final result length: {len(res)}")

        return res

    def _get_cross_encoder_scores(
        self, utterances: list[str], candidates: list[list[str]]
    ):
        """
        Arguments
        ---
        `utterances`: list of query utterances
        `candidates`: for each query, this list contains a list of the k the closest sample utterances (from retrieval module)

        Return
        ---
        for each query, return a list of a corresponding cross encoder scores for the k the closest sample utterances
        """
        assert len(utterances) == len(candidates)

        text_pairs = [
            [[query, cand] for cand in docs]
            for query, docs in zip(utterances, candidates)
        ]

        flattened_text_pairs = list(it.chain.from_iterable(text_pairs))

        assert len(flattened_text_pairs) == len(utterances) * len(candidates[0])

        flattened_cross_encoder_scores = self.model.predict(flattened_text_pairs)
        cross_encoder_scores = [
            flattened_cross_encoder_scores[i : i + self.k]
            for i in range(0, len(flattened_cross_encoder_scores), self.k)
        ]
        return cross_encoder_scores

    def _build_result(self, scores: list[list[float]], labels: list[list[int]]):
        logger.info(f"Input scores shape: {np.array(scores).shape}")
        logger.info(f"Input labels shape: {np.array(labels).shape}")
        logger.info(f"Sample scores: {scores[:2]}")
        logger.info(f"Sample labels: {labels[:2]}")

        scores = np.array(scores)
        labels = np.array(labels, dtype=int)
        n_classes = self._collection.metadata["n_classes"]

        logger.info(f"n_classes: {n_classes}")
        logger.info(f"Processed scores shape: {scores.shape}")
        logger.info(f"Processed labels shape: {labels.shape}")

        return build_result(scores, labels, n_classes)

    def clear_cache(self):
        model = self._collection._embedding_function._model
        model.to(device='cpu')
        del model
        self._collection = None


def build_result(scores: np.ndarray, labels: np.ndarray, n_classes: int):
    logger.info(f"build_result input shapes - scores: {scores.shape}, labels: {labels.shape}")

    res = np.zeros((len(scores), n_classes))

    best_neighbors = np.argmax(scores, axis=1)
    idx_helper = np.arange(len(res))
    best_classes = labels[idx_helper, best_neighbors]
    best_scores = scores[idx_helper, best_neighbors]

    logger.info(f"Best neighbors shape: {best_neighbors.shape}")
    logger.info(f"Best classes shape: {best_classes.shape}")
    logger.info(f"Best scores shape: {best_scores.shape}")

    res[idx_helper, best_classes] = best_scores

    return res

