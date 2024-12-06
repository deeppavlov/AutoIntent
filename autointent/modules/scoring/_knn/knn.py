"""KNNScorer class for k-nearest neighbors scoring."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from autointent.context import Context
from autointent.context.vector_index_client import VectorIndex, VectorIndexClient, get_db_dir
from autointent.custom_types import WEIGHT_TYPES, BaseMetadataDict, LabelType
from autointent.modules.abc import ScoringModule

from .weighting import apply_weights


class KNNScorerDumpMetadata(BaseMetadataDict):
    """
    Metadata for dumping the state of a KNNScorer.

    :ivar n_classes: Number of classes in the dataset.
    :ivar multilabel: Whether the task is multilabel classification.
    :ivar db_dir: Path to the database directory.
    :ivar batch_size: Batch size used for embedding.
    :ivar max_length: Maximum sequence length for embedding, or None if not specified.
    """

    n_classes: int
    multilabel: bool
    db_dir: str
    batch_size: int
    max_length: int | None


class KNNScorer(ScoringModule):
    """
    K-nearest neighbors (KNN) scorer for intent classification.

    This module uses a vector index to retrieve nearest neighbors for query utterances
    and applies a weighting strategy to compute class probabilities.

    :ivar weights: Weighting strategy used for scoring.
    :ivar _vector_index: VectorIndex instance for neighbor retrieval.
    :ivar name: Name of the scorer, defaults to "knn".
    :ivar prebuilt_index: Flag indicating if the vector index is prebuilt.

    Examples
    --------
    .. testcode::

        from autointent.modules.scoring import KNNScorer
        utterances = ["hello", "how are you?"]
        labels = [0, 1]
        scorer = KNNScorer(
            embedder_name="sergeyzh/rubert-tiny-turbo",
            k=5,
        )
        scorer.fit(utterances, labels)
        test_utterances = ["hi", "what's up?"]
        probabilities = scorer.predict(test_utterances)
        print(probabilities)  # Outputs predicted class probabilities for the utterances

    .. testoutput::

        [[0.67297815 0.32702185]
         [0.44031678 0.55968322]]

    """

    weights: WEIGHT_TYPES
    _vector_index: VectorIndex
    name = "knn"
    prebuilt_index: bool = False
    max_length: int | None

    def __init__(
        self,
        embedder_name: str,
        k: int,
        weights: WEIGHT_TYPES = "distance",
        db_dir: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int | None = None,
        embedder_use_cache: bool = False,
    ) -> None:
        """
        Initialize the KNNScorer.

        :param embedder_name: Name of the embedder used for vectorization.
        :param k: Number of closest neighbors to consider during inference.
        :param weights: Weighting strategy:
            - "uniform" (or False): Equal weight for all neighbors.
            - "distance" (or True): Weight inversely proportional to distance.
            - "closest": Only the closest neighbor of each class is weighted.
        :param db_dir: Path to the database directory, or None to use default.
        :param device: Device to run operations on, e.g., "cpu" or "cuda".
        :param batch_size: Batch size for embedding generation, defaults to 32.
        :param max_length: Maximum sequence length for embedding, or None for default.
        :param embedder_use_cache: Flag indicating whether to cache intermediate embeddings.
        """
        self.embedder_name = embedder_name
        self.k = k
        self.weights = weights
        self._db_dir = db_dir
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedder_use_cache = embedder_use_cache

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
        k: int,
        weights: WEIGHT_TYPES,
        embedder_name: str | None = None,
    ) -> "KNNScorer":
        """
        Create a KNNScorer instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param k: Number of closest neighbors to consider during inference.
        :param weights: Weighting strategy for scoring.
        :param embedder_name: Name of the embedder, or None to use the best embedder.
        :return: Initialized KNNScorer instance.
        """
        if embedder_name is None:
            embedder_name = context.optimization_info.get_best_embedder()
            prebuilt_index = True
        else:
            prebuilt_index = context.vector_index_client.exists(embedder_name)

        instance = cls(
            embedder_name=embedder_name,
            k=k,
            weights=weights,
            db_dir=str(context.get_db_dir()),
            device=context.get_device(),
            batch_size=context.get_batch_size(),
            max_length=context.get_max_length(),
            embedder_use_cache=context.get_use_cache(),
        )
        instance.prebuilt_index = prebuilt_index
        return instance

    def get_embedder_name(self) -> str:
        """
        Get the name of the embedder.

        :return: Embedder name.
        """
        return self.embedder_name

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        """
        Fit the scorer by training or loading the vector index.

        :param utterances: List of training utterances.
        :param labels: List of labels corresponding to the utterances.
        :raises ValueError: If the vector index mismatches the provided utterances.
        """
        if isinstance(labels[0], list):
            self.n_classes = len(labels[0])
            self.multilabel = True
        else:
            self.n_classes = len(set(labels))
            self.multilabel = False
        vector_index_client = VectorIndexClient(self.device, self.db_dir, embedder_use_cache=self.embedder_use_cache)

        if self.prebuilt_index:
            # this happens only after RetrievalNode optimization
            self._vector_index = vector_index_client.get_index(self.embedder_name)
            if len(utterances) != len(self._vector_index.texts):
                msg = "Vector index mismatches provided utterances"
                raise ValueError(msg)
        else:
            self._vector_index = vector_index_client.create_index(self.embedder_name, utterances, labels)

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        """
        Predict class probabilities for the given utterances.

        :param utterances: List of query utterances.
        :return: Array of predicted probabilities for each class.
        """
        return self._predict(utterances)[0]

    def predict_with_metadata(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[dict[str, Any]] | None]:
        """
        Predict class probabilities along with metadata for the given utterances.

        :param utterances: List of query utterances.
        :return: Tuple of predicted probabilities and metadata with neighbor information.
        """
        scores, neighbors = self._predict(utterances)
        metadata = [{"neighbors": utterance_neighbors} for utterance_neighbors in neighbors]
        return scores, metadata

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the vector index."""
        self._vector_index.clear_ram()

    def dump(self, path: str) -> None:
        """
        Save the KNNScorer's metadata and vector index to disk.

        :param path: Path to the directory where assets will be dumped.
        """
        self.metadata = self._store_state_to_metadata()

        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

        self._vector_index.dump(dump_dir)

    def _store_state_to_metadata(self) -> KNNScorerDumpMetadata:
        return KNNScorerDumpMetadata(
            db_dir=self.db_dir,
            n_classes=self.n_classes,
            multilabel=self.multilabel,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

    def load(self, path: str) -> None:
        """
        Load the KNNScorer's metadata and vector index from disk.

        :param path: Path to the directory containing the dumped assets.
        """
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            self.metadata: KNNScorerDumpMetadata = json.load(file)

        self._restore_state_from_metadata(self.metadata)

    def _restore_state_from_metadata(self, metadata: KNNScorerDumpMetadata) -> None:
        self.n_classes = metadata["n_classes"]
        self.multilabel = metadata["multilabel"]

        vector_index_client = VectorIndexClient(
            device=self.device,
            db_dir=metadata["db_dir"],
            embedder_batch_size=metadata["batch_size"],
            embedder_max_length=metadata["max_length"],
            embedder_use_cache=self.embedder_use_cache,
        )
        self._vector_index = vector_index_client.get_index(self.embedder_name)

    def _get_neighbours(
        self, utterances: list[str]
    ) -> tuple[list[list[LabelType]], list[list[float]], list[list[str]]]:
        return self._vector_index.query(utterances, self.k)

    def _count_scores(self, labels: npt.NDArray[Any], distances: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return apply_weights(labels, distances, self.weights, self.n_classes, self.multilabel)

    def _predict(self, utterances: list[str]) -> tuple[npt.NDArray[Any], list[list[str]]]:
        """
        Predict class probabilities and retrieve neighbors for the given utterances.

        :param utterances: List of query utterances.
        :return: Tuple containing class probabilities and neighbor utterances.
        """
        labels, distances, neighbors = self._get_neighbours(utterances)
        scores = self._count_scores(np.array(labels), np.array(distances))
        return scores, neighbors
