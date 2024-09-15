import itertools as it

import numpy as np
from sentence_transformers import CrossEncoder
from ..base import Context, ScoringModule
from .head_training import CrossEncoderWithLogreg

class DNNCScorer(ScoringModule):

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

        query_res = self._collection.query(
            query_texts=utterances,
            n_results=self.k,
            include=["metadatas", "documents"],  # one can add "embeddings", "distances"
        )

        cross_encoder_scores = self._get_cross_encoder_scores(
            utterances, query_res["documents"]
        )

        labels_pred = []
        for i, candidates in enumerate(query_res["metadatas"]):
            try:
                intent_ids = [int(next((k for k, v in cand.items() if v == 1), -1)) for cand in candidates]
                labels_pred.append(intent_ids)
            except Exception as e:
                raise
        res = self._build_result(cross_encoder_scores, labels_pred)

        return res

    def _get_cross_encoder_scores(
        self, utterances: list[str], candidates: list[list[str]]
    ):
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
        scores = np.array(scores)
        labels = np.array(labels, dtype=int)
        n_classes = self._collection.metadata["n_classes"]
        return build_result(scores, labels, n_classes)

    def clear_cache(self):
        model = self._collection._embedding_function._model
        model.to(device='cpu')
        del model
        self._collection = None


def build_result(scores: np.ndarray, labels: np.ndarray, n_classes: int):
    res = np.zeros((len(scores), n_classes))

    for i in range(len(scores)):
        for j in range(len(scores[i])):
            class_label = labels[i, j]
            res[i, class_label] = max(res[i, class_label], scores[i, j])

    return res

