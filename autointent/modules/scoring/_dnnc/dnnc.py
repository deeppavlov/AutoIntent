"""DNNCScorer class for scoring utterances using deep neural network classifiers (DNNC)."""

import itertools as it
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sentence_transformers import CrossEncoder

from autointent import Context, VectorIndex
from autointent.custom_types import BaseMetadataDict, LabelType
from autointent.modules.abc import ScoringModule

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
    :ivar _db_dir: Path to the database directory where the vector index is stored.
    :ivar name: Name of the scorer, defaults to "dnnc".

    Examples
    --------
    .. testsetup::

        db_dir = "doctests-db"

    .. testcode::

        from autointent.modules.scoring import DNNCScorer
        utterances = ["what is your name?", "how are you?"]
        labels = [0, 1]
        scorer = DNNCScorer(
            cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            embedder_name="sergeyzh/rubert-tiny-turbo",
            k=5,
            db_dir=db_dir,
        )
        scorer.fit(utterances, labels)

        test_utterances = ["Hello!", "What's up?"]
        scores = scorer.predict(test_utterances)
        print(scores)  # Outputs similarity scores for the utterances


    .. testoutput::

        [[-8.90408421  0.        ]
         [-8.10923195  0.        ]]

    .. testcleanup::

        import shutil
        shutil.rmtree(db_dir)

    """

    name = "dnnc"

    crossencoder_subdir: str = "crossencoder"
    model: CrossEncoder | CrossEncoderWithLogreg

    def __init__(
        self,
        cross_encoder_name: str,
        embedder_name: str,
        k: int,
        device: str = "cpu",
        train_head: bool = False,
        batch_size: int = 32,
        max_length: int | None = None,
        embedder_use_cache: bool = True,
    ) -> None:
        """
        Initialize the DNNCScorer.

        :param cross_encoder_name: Name of the cross-encoder model.
        :param embedder_name: Name of the embedder model.
        :param k: Number of nearest neighbors to retrieve.
        :param device: Device to run operations on, e.g., "cpu" or "cuda".
        :param train_head: Whether to train a logistic regression head, defaults to False.
        :param batch_size: Batch size for processing text pairs, defaults to 32.
        :param max_length: Maximum sequence length for embedding, or None for default.
        :param embedder_use_cache: Flag indicating whether to cache intermediate embeddings.
        """
        self.cross_encoder_name = cross_encoder_name
        self.embedder_name = embedder_name
        self.k = k
        self.train_head = train_head
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedder_use_cache = embedder_use_cache

    @classmethod
    def from_context(
        cls,
        context: Context,
        cross_encoder_name: str,
        k: int,
        embedder_name: str | None = None,
        train_head: bool = False,
    ) -> "DNNCScorer":
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

        return cls(
            cross_encoder_name=cross_encoder_name,
            embedder_name=embedder_name,
            k=k,
            train_head=train_head,
            device=context.get_device(),
            batch_size=context.get_batch_size(),
            max_length=context.get_max_length(),
            embedder_use_cache=context.get_use_cache(),
        )

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        """
        Fit the scorer by training or loading the vector index and optionally training a logistic regression head.

        :param utterances: List of training utterances.
        :param labels: List of labels corresponding to the utterances.
        :raises ValueError: If the vector index mismatches the provided utterances.
        """
        self.n_classes = len(set(labels))

        self.model = CrossEncoder(self.cross_encoder_name, trust_remote_code=True, device=self.device)

        self.vector_index = VectorIndex(
            self.embedder_name,
            self.device,
            self.batch_size,
            self.max_length,
            self.embedder_use_cache,
        )
        self.vector_index.add(utterances, labels)

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
            flattened_cross_encoder_scores[i : i + self.k].tolist()  # type: ignore[misc]
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
