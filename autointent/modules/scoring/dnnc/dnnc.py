import itertools as it
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sentence_transformers import CrossEncoder

from autointent import Context
from autointent.context.vector_index_client import VectorIndexClient
from autointent.modules.scoring.base import ScoringModule

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
        self.n_classes = context.n_classes
        self.model = CrossEncoder(self.model_name, trust_remote_code=True, device=context.device)
        self.vector_index = context.get_best_index()

        if self.train_head:
            model = CrossEncoderWithLogreg(self.model)
            model.fit(context.data_handler.utterances_train, context.data_handler.labels_train)
            self.model = model

        self.metadata = {
            "device": context.device,
            "db_dir": context.db_dir,
            "n_classes": self.n_classes,
            "biencoder_model": self.vector_index.model_name,
        }

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """
        Return
        ---
        `(n_queries, n_classes)` matrix with zeros everywhere except the class of the best neighbor utterance
        """
        labels, _, texts = self.vector_index.query(
            utterances,
            self.k,
        )

        cross_encoder_scores = self._get_cross_encoder_scores(utterances, texts)

        return self._build_result(cross_encoder_scores, labels)

    def _get_cross_encoder_scores(self, utterances: list[str], candidates: list[list[str]]) -> list[list[float]]:
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

    def _build_result(self, scores: list[list[float]], labels: list[list[int]]) -> npt.NDArray[Any]:
        """
        Arguments
        ---
        `scores`: for each query utterance, cross encoder scores of its k closest utterances
        `labels`: corresponding intent labels

        Return
        ---
        `(n_queries, n_classes)` matrix with zeros everywhere except the class of the best neighbor utterance
        """
        n_classes = self.n_classes

        return build_result(np.array(scores), np.array(labels), n_classes)

    def clear_cache(self) -> None:
        pass

    def dump(self, path: str) -> None:
        dump_dir = Path(path)
        with (dump_dir / "metadata.json").open("w") as file:
            json.dump(self.metadata, file, indent=4)

        crossencoder_dir = str(dump_dir / "crossencoder")
        self.model.save(crossencoder_dir)

    def load(self, path: str) -> None:
        dump_dir = Path(path)
        with (dump_dir / "metadata.json").open() as file:
            self.metadata = json.load(file)

        self.n_classes = self.metadata["n_classes"]

        vector_index_client = VectorIndexClient(device=self.metadata["device"], db_dir=self.metadata["db_dir"])
        self.vector_index = vector_index_client.get_index(self.metadata["biencoder_model"])

        crossencoder_dir = str(dump_dir / "crossencoder")
        if not self.train_head:
            self.model = CrossEncoder(crossencoder_dir, device=self.metadata["device"])
        else:
            self.model = CrossEncoderWithLogreg.load(crossencoder_dir)


def build_result(scores: npt.NDArray[Any], labels: npt.NDArray[Any], n_classes: int) -> npt.NDArray[Any]:
    res = np.zeros((len(scores), n_classes))
    best_neighbors = np.argmax(scores, axis=1)
    idx_helper = np.arange(len(res))
    best_classes = labels[idx_helper, best_neighbors]
    best_scores = scores[idx_helper, best_neighbors]
    res[idx_helper, best_classes] = best_scores
    return res
