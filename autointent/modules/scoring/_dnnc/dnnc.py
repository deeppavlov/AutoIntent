"""DNNCScorer class for scoring utterances using deep neural network classifiers (DNNC)."""

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
from autointent.context.vector_index_client import VectorIndexClient, get_db_dir
from autointent.custom_types import BaseMetadataDict, LabelType
from autointent.modules.scoring import ScoringModule

from .head_training import CrossEncoderWithLogreg

logger = logging.getLogger(__name__)


class DNNCScorerDumpMetadata(BaseMetadataDict):
    """Metadata for dumping the state of a DNNCScorer."""

    db_dir: str
    n_classes: int
    batch_size: int
    max_length: int | None


class DNNCScorer(ScoringModule):
    r"""
    Scoring module for intent classification using a discriminative nearest neighbor classification (DNNC).

    This module uses a CrossEncoder for scoring candidate intents and can optionally
    train a logistic regression head on top of cross-encoder features.

    .. code-block:: bibtex

        @misc{zhang2020discriminativenearestneighborfewshot,
          title={Discriminative Nearest Neighbor Few-Shot Intent Detection by Transferring Natural Language Inference},
          author={Jian-Guo Zhang and Kazuma Hashimoto and Wenhao Liu and Chien-Sheng Wu and Yao Wan and
          Philip S. Yu and Richard Socher and Caiming Xiong},
          year={2020},
          eprint={2010.13009},
          archivePrefix={arXiv},
          primaryClass={cs.CL},
          url={https://arxiv.org/abs/2010.13009},
        }

    :ivar crossencoder_subdir: Subdirectory for storing the cross-encoder model (`crossencoder`).
    :ivar model: The model used for scoring, which could be a `CrossEncoder` or a `CrossEncoderWithLogreg`.
    :ivar prebuilt_index: Flag indicating whether a prebuilt vector index is used.
    :ivar _db_dir: Path to the database directory where the vector index is stored.
    :ivar name: Name of the scorer, defaults to "dnnc".

    Examples
    --------
    Creating and fitting the DNNCScorer:
    >>> from autointent.modules import DNNCScorer
    >>> utterances = ["what is your name?", "how are you?"]
    >>> labels = ["greeting", "greeting"]
    >>> scorer = DNNCScorer(
    >>>     cross_encoder_name="cross_encoder_model",
    >>>     embedder_name="embedder_model",
    >>>     k=5,
    >>>     db_dir="/path/to/database",
    >>>     device="cuda",
    >>>     train_head=True,
    >>>     batch_size=32,
    >>>     max_length=128
    >>> )
    >>> scorer.fit(utterances, labels)

    Predicting scores:
    >>> test_utterances = ["Hello!", "What's up?"]
    >>> scores = scorer.predict(test_utterances)
    >>> print(scores)  # Outputs similarity scores for the utterances

    Saving and loading the scorer:
    >>> scorer.dump("outputs/")
    >>> loaded_scorer = DNNCScorer(
    >>>     cross_encoder_name="cross_encoder_model",
    >>>     embedder_name="embedder_model",
    >>>     k=5,
    >>>     db_dir="/path/to/database",
    >>>     device="cuda"
    >>> )
    >>> loaded_scorer.load("outputs/")
    """

    name = "dnnc"

    crossencoder_subdir: str = "crossencoder"
    model: CrossEncoder | CrossEncoderWithLogreg
    prebuilt_index: bool = False

    def __init__(
        self,
        cross_encoder_name: str,
        embedder_name: str,
        k: int,
        db_dir: str | None = None,
        device: str = "cpu",
        train_head: bool = False,
        batch_size: int = 32,
        max_length: int | None = None,
    ) -> None:
        """
        Initialize the DNNCScorer.

        :param cross_encoder_name: Name of the cross-encoder model.
        :param embedder_name: Name of the embedder model.
        :param k: Number of nearest neighbors to retrieve.
        :param db_dir: Path to the database directory, or None to use default.
        :param device: Device to run operations on, e.g., "cpu" or "cuda".
        :param train_head: Whether to train a logistic regression head, defaults to False.
        :param batch_size: Batch size for processing text pairs, defaults to 32.
        :param max_length: Maximum sequence length for embedding, or None for default.
        """
        self.cross_encoder_name = cross_encoder_name
        self.embedder_name = embedder_name
        self.k = k
        self.train_head = train_head
        self.device = device
        self._db_dir = db_dir
        self.batch_size = batch_size
        self.max_length = max_length

    @property
    def db_dir(self) -> str:
        """
        Get the database directory for the vector index.

        :return: Path to the database directory.
        """
        if self._db_dir is None:
            self._db_dir = str(get_db_dir())
        return self._db_dir

    @classmethod
    def from_context(
        cls,
        context: Context,
        cross_encoder_name: str,
        k: int,
        embedder_name: str | None = None,
        train_head: bool = False,
    ) -> Self:
        """
        Create a DNNCScorer instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param cross_encoder_name: Name of the cross-encoder model.
        :param k: Number of nearest neighbors to retrieve.
        :param embedder_name: Name of the embedder model, or None to use the best embedder.
        :param train_head: Whether to train a logistic regression head, defaults to False.
        :return: Initialized DNNCScorer instance.
        """
        if embedder_name is None:
            embedder_name = context.optimization_info.get_best_embedder()
            prebuilt_index = True
        else:
            prebuilt_index = context.vector_index_client.exists(embedder_name)

        instance = cls(
            cross_encoder_name=cross_encoder_name,
            embedder_name=embedder_name,
            k=k,
            train_head=train_head,
            device=context.get_device(),
            db_dir=str(context.get_db_dir()),
            batch_size=context.get_batch_size(),
            max_length=context.get_max_length(),
        )
        instance.prebuilt_index = prebuilt_index
        return instance

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        """
        Fit the scorer by training or loading the vector index and optionally training a logistic regression head.

        :param utterances: List of training utterances.
        :param labels: List of labels corresponding to the utterances.
        :raises ValueError: If the vector index mismatches the provided utterances.
        """
        self.n_classes = len(set(labels))

        self.model = CrossEncoder(self.cross_encoder_name, trust_remote_code=True, device=self.device)

        vector_index_client = VectorIndexClient(self.device, self.db_dir)

        if self.prebuilt_index:
            # this happens only when LinearScorer is within Pipeline opimization after RetrievalNode optimization
            self.vector_index = vector_index_client.get_index(self.embedder_name)
            if len(utterances) != len(self.vector_index.texts):
                msg = "Vector index mismatches provided utterances"
                raise ValueError(msg)
        else:
            self.vector_index = vector_index_client.create_index(self.embedder_name, utterances, labels)

        if self.train_head:
            model = CrossEncoderWithLogreg(self.model)
            model.fit(utterances, labels)
            self.model = model

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """
        Predict class scores for the given utterances.

        :param utterances: List of utterances to score.
        :return: Array of predicted scores.
        """
        return self._predict(utterances)[0]

    def predict_with_metadata(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[dict[str, Any]] | None]:
        """
        Predict class scores along with metadata for the given utterances.

        :param utterances: List of utterances to score.
        :return: Tuple of scores and metadata containing neighbor details and scores.
        """
        scores, neighbors, neighbors_scores = self._predict(utterances)
        metadata = [
            {"neighbors": utterance_neighbors, "scores": utterance_neighbors_scores}
            for utterance_neighbors, utterance_neighbors_scores in zip(neighbors, neighbors_scores, strict=True)
        ]
        return scores, metadata

    def _get_cross_encoder_scores(self, utterances: list[str], candidates: list[list[str]]) -> list[list[float]]:
        """
        Compute cross-encoder scores for utterances against their candidate neighbors.

        :param utterances: List of query utterances.
        :param candidates: List of candidate utterances for each query.
        :return: List of cross-encoder scores for each query-candidate pair.
        :raises ValueError: If the number of utterances and candidates do not match.
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

    def _build_result(self, scores: list[list[float]], labels: list[list[LabelType]]) -> npt.NDArray[Any]:
        """
        Build a result matrix with scores assigned to the best neighbor's class.

        :param scores: for each query utterance, cross encoder scores of its k closest utterances
        :param labels: corresponding intent labels

        :return: (n_queries, n_classes) matrix with zeros everywhere except the class of the best neighbor utterance
        """
        n_classes = self.n_classes

        return build_result(np.array(scores), np.array(labels), n_classes)

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the vector index."""
        self.vector_index.clear_ram()

    def dump(self, path: str) -> None:
        """
        Save the DNNCScorer's metadata, vector index, and model to disk.

        :param path: Path to the directory where assets will be dumped.
        """
        self.metadata = DNNCScorerDumpMetadata(
            db_dir=self.db_dir,
            n_classes=self.n_classes,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        dump_dir = Path(path)
        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

        crossencoder_dir = str(dump_dir / self.crossencoder_subdir)
        self.model.save(crossencoder_dir)
        self.vector_index.dump(Path(self.db_dir))

    def load(self, path: str) -> None:
        """
        Load the DNNCScorer's metadata, vector index, and model from disk.

        :param path: Path to the directory containing the dumped assets.
        """
        dump_dir = Path(path)
        with (dump_dir / self.metadata_dict_name).open() as file:
            self.metadata: DNNCScorerDumpMetadata = json.load(file)

        self.n_classes = self.metadata["n_classes"]

        vector_index_client = VectorIndexClient(
            device=self.device,
            db_dir=self.metadata["db_dir"],
            embedder_batch_size=self.metadata["batch_size"],
            embedder_max_length=self.metadata["max_length"],
        )
        self.vector_index = vector_index_client.get_index(self.embedder_name)

        crossencoder_dir = str(dump_dir / self.crossencoder_subdir)
        if self.train_head:
            self.model = CrossEncoderWithLogreg.load(crossencoder_dir)
        else:
            self.model = CrossEncoder(crossencoder_dir, device=self.device)

    def _predict(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[list[str]], list[list[float]]]:
        """
        Predict class scores for the given utterances using the vector index and cross-encoder.

        :param utterances: List of query utterances.
        :return: Tuple containing class scores, neighbor utterances, and neighbor scores.
        """
        labels, _, neighbors = self.vector_index.query(
            utterances,
            self.k,
        )

        cross_encoder_scores = self._get_cross_encoder_scores(utterances, neighbors)

        return self._build_result(cross_encoder_scores, labels), neighbors, cross_encoder_scores


def build_result(scores: npt.NDArray[Any], labels: npt.NDArray[Any], n_classes: int) -> npt.NDArray[Any]:
    """
    Build a result matrix with scores assigned to the best neighbor's class.

    :param scores: Cross-encoder scores for each query's neighbors.
    :param labels: Labels corresponding to each neighbor.
    :param n_classes: Total number of classes.
    :return: Matrix of size (n_queries, n_classes) with scores for the best class.
    """
    res = np.zeros((len(scores), n_classes))
    best_neighbors = np.argmax(scores, axis=1)
    idx_helper = np.arange(len(res))
    best_classes = labels[idx_helper, best_neighbors]
    best_scores = scores[idx_helper, best_neighbors]
    res[idx_helper, best_classes] = best_scores
    return res
