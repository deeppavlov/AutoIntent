import itertools as it
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sentence_transformers import CrossEncoder
from typing_extensions import Self

from autointent import Context
from autointent.context.vector_index_client import VectorIndexClient
from autointent.context.vector_index_client.cache import get_db_dir
from autointent.custom_types import LABEL_TYPE, BaseMetadataDict
from autointent.modules.scoring.base import ScoringModule

from .head_training import CrossEncoderWithLogreg

logger = logging.getLogger(__name__)


class DNNCScorerDumpMetadata(BaseMetadataDict):
    device: str
    db_dir: str
    n_classes: int
    cross_encoder_name: str
    search_model_name: str
    k: int
    train_head: bool


class DNNCScorer(ScoringModule):
    """
    TODO:
    - think about other cross-encoder settings
    - implement training of cross-encoder with sentence_encoders utils
    - inspect batch size of model.predict?
    """

    crossencoder_subdir: str = "crossencoder"
    model: CrossEncoder | CrossEncoderWithLogreg
    prebuilt_index: bool = False

    def __init__(
        self,
        cross_encoder_name: str,
        search_model_name: str,
        k: int,
        db_dir: str | None = None,
        device: str = "cpu",
        train_head: bool = False,
    ) -> None:
        if db_dir is None:
            db_dir = str(get_db_dir())

        self.cross_encoder_name = cross_encoder_name
        self.search_model_name = search_model_name
        self.k = k
        self.train_head = train_head
        self.device = device
        self.db_dir = db_dir

    @classmethod
    def from_context(
        cls,
        context: Context,
        cross_encoder_name: str,
        k: int,
        search_model_name: str | None = None,
        train_head: bool = False,
    ) -> Self:
        prebuilt_index = False
        if search_model_name is None:
            search_model_name = context.optimization_info.get_best_embedder()
            prebuilt_index = True
        instance = cls(
            cross_encoder_name=cross_encoder_name,
            search_model_name=search_model_name,
            k=k,
            train_head=train_head,
            device=context.device,
            db_dir=str(context.db_dir),
        )
        instance.prebuilt_index = prebuilt_index
        return instance

    def fit(self, utterances: list[str], labels: list[LABEL_TYPE], **kwargs: dict[str, Any]) -> None:
        self.n_classes = len(set(labels))

        self.model = CrossEncoder(self.cross_encoder_name, trust_remote_code=True, device=self.device)

        vector_index_client = VectorIndexClient(self.device, self.db_dir)

        if self.prebuilt_index:
            # this happens only when LinearScorer is within Pipeline opimization after RetrievalNode optimization
            self.vector_index = vector_index_client.get_index(self.search_model_name)
            if len(utterances) != len(self.vector_index.texts):
                msg = "Vector index mismatches provided utterances"
                raise ValueError(msg)
        else:
            self.vector_index = vector_index_client.create_index(self.search_model_name, utterances, labels)

        if self.train_head:
            model = CrossEncoderWithLogreg(self.model)
            model.fit(utterances, labels)
            self.model = model

        self.metadata = DNNCScorerDumpMetadata(
            device=self.device,
            db_dir=self.db_dir,
            n_classes=self.n_classes,
            cross_encoder_name=self.cross_encoder_name,
            search_model_name=self.search_model_name,
            k=self.k,
            train_head=self.train_head,
        )

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

        flattened_cross_encoder_scores: npt.NDArray[np.float64] = self.model.predict(flattened_text_pairs)  # type: ignore[assignment]
        return [
            flattened_cross_encoder_scores[i : i + self.k].tolist()
            for i in range(0, len(flattened_cross_encoder_scores), self.k)
        ]

    def _build_result(self, scores: list[list[float]], labels: list[list[LABEL_TYPE]]) -> npt.NDArray[Any]:
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
        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

        crossencoder_dir = str(dump_dir / self.crossencoder_subdir)
        self.model.save(crossencoder_dir)
        self.vector_index.dump(Path(self.db_dir))

    def load(self, path: str) -> None:
        dump_dir = Path(path)
        with (dump_dir / self.metadata_dict_name).open() as file:
            self.metadata: DNNCScorerDumpMetadata = json.load(file)

        self.n_classes = self.metadata["n_classes"]
        self.cross_encoder_name = self.metadata["cross_encoder_name"]
        self.search_model_name = self.metadata["search_model_name"]
        self.train_head = self.metadata["train_head"]

        vector_index_client = VectorIndexClient(device=self.metadata["device"], db_dir=self.metadata["db_dir"])
        self.vector_index = vector_index_client.get_index(self.search_model_name)

        crossencoder_dir = str(dump_dir / self.crossencoder_subdir)
        if self.train_head:
            self.model = CrossEncoderWithLogreg.load(crossencoder_dir)
        else:
            self.model = CrossEncoder(crossencoder_dir, device=self.metadata["device"])


def build_result(scores: npt.NDArray[Any], labels: npt.NDArray[Any], n_classes: int) -> npt.NDArray[Any]:
    res = np.zeros((len(scores), n_classes))
    best_neighbors = np.argmax(scores, axis=1)
    idx_helper = np.arange(len(res))
    best_classes = labels[idx_helper, best_neighbors]
    best_scores = scores[idx_helper, best_neighbors]
    res[idx_helper, best_classes] = best_scores
    return res
