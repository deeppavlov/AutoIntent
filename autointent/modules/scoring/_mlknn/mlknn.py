"""MLKnnScorer class for multi-label k-nearest neighbors classification."""

import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from autointent import Context
from autointent.context.vector_index_client import VectorIndexClient, get_db_dir
from autointent.custom_types import BaseMetadataDict, LabelType
from autointent.modules.scoring import ScoringModule


class MLKnnScorerDumpMetadata(BaseMetadataDict):
    """
    Metadata for dumping the state of an MLKnnScorer.

    :ivar db_dir: Path to the database directory.
    :ivar n_classes: Number of classes in the dataset.
    :ivar batch_size: Batch size used for embedding.
    :ivar max_length: Maximum sequence length for embedding, or None if not specified.
    """

    db_dir: str
    n_classes: int
    batch_size: int
    max_length: int | None


class ArrayToSave(TypedDict):
    """
    Data structure for saving prior and conditional probabilities.

    :ivar prior_prob_true: Prior probabilities of each class being true.
    :ivar prior_prob_false: Prior probabilities of each class being false.
    :ivar cond_prob_true: Conditional probabilities given true labels.
    :ivar cond_prob_false: Conditional probabilities given false labels.
    """

    prior_prob_true: NDArray[np.float64]
    prior_prob_false: NDArray[np.float64]
    cond_prob_true: NDArray[np.float64]
    cond_prob_false: NDArray[np.float64]


class MLKnnScorer(ScoringModule):
    """
    Multi-label k-nearest neighbors (ML-KNN) scorer.

    This module implements ML-KNN, a multi-label classifier that computes probabilities
    based on the k-nearest neighbors of a query instance.

    :ivar arrays_filename: Filename for saving probabilities to disk.
    :ivar metadata: Metadata about the scorer's configuration.
    :ivar prebuilt_index: Flag indicating if the vector index is prebuilt.
    :ivar name: Name of the scorer, defaults to "mlknn".

    Example
    --------
    Creating and fitting the MLKnnScorer:
    >>> from knn_scorer import MLKnnScorer
    >>> utterances = ["what is your name?", "how are you?"]
    >>> labels = [["greeting"], ["greeting"]]
    >>> scorer = MLKnnScorer(
    >>>     k=5,
    >>>     embedder_name="bert-base",
    >>>     db_dir="/path/to/database",
    >>>     s=1.0,
    >>>     ignore_first_neighbours=0,
    >>>     device="cuda",
    >>>     batch_size=32,
    >>>     max_length=128
    >>> )
    >>> scorer.fit(utterances, labels)

    Predicting probabilities:
    >>> test_utterances = ["Hi!", "What's up?"]
    >>> probabilities = scorer.predict(test_utterances)
    >>> print(probabilities)  # Outputs predicted probabilities for each label

    Predicting labels:
    >>> predicted_labels = scorer.predict_labels(test_utterances, thresh=0.5)
    >>> print(predicted_labels)  # Outputs binary array for each label prediction

    Saving and loading the scorer:
    >>> scorer.dump("outputs/")
    >>> loaded_scorer = MLKnnScorer(
    >>>     k=5,
    >>>     embedder_name="bert-base",
    >>>     db_dir="/path/to/database",
    >>>     device="cuda"
    >>> )
    >>> loaded_scorer.load("outputs/")
    """

    arrays_filename: str = "probs.npz"
    metadata: MLKnnScorerDumpMetadata
    prebuilt_index: bool = False
    name = "mlknn"

    def __init__(
        self,
        k: int,
        embedder_name: str,
        db_dir: str | None = None,
        s: float = 1.0,
        ignore_first_neighbours: int = 0,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int | None = None,
    ) -> None:
        """
        Initialize the MLKnnScorer.

        :param k: Number of nearest neighbors to consider.
        :param embedder_name: Name of the embedder used for vectorization.
        :param db_dir: Path to the database directory, or None to use default.
        :param s: Smoothing parameter for probability calculations, defaults to 1.0.
        :param ignore_first_neighbours: Number of closest neighbors to ignore, defaults to 0.
        :param device: Device to run operations on, e.g., "cpu" or "cuda".
        :param batch_size: Batch size for embedding generation, defaults to 32.
        :param max_length: Maximum sequence length for embedding, or None for default.
        """
        self.k = k
        self.embedder_name = embedder_name
        self.s = s
        self.ignore_first_neighbours = ignore_first_neighbours
        self._db_dir = db_dir
        self.device = device
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
        k: int,
        s: float = 1.0,
        ignore_first_neighbours: int = 0,
        embedder_name: str | None = None,
    ) -> "MLKnnScorer":
        """
        Create an MLKnnScorer instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param k: Number of nearest neighbors to consider.
        :param s: Smoothing parameter for probability calculations, defaults to 1.0.
        :param ignore_first_neighbours: Number of closest neighbors to ignore, defaults to 0.
        :param embedder_name: Name of the embedder, or None to use the best embedder.
        :return: Initialized MLKnnScorer instance.
        """
        if embedder_name is None:
            embedder_name = context.optimization_info.get_best_embedder()
            prebuilt_index = True
        else:
            prebuilt_index = context.vector_index_client.exists(embedder_name)

        instance = cls(
            k=k,
            embedder_name=embedder_name,
            s=s,
            ignore_first_neighbours=ignore_first_neighbours,
            db_dir=str(context.get_db_dir()),
            device=context.get_device(),
            batch_size=context.get_batch_size(),
            max_length=context.get_max_length(),
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
        Fit the scorer by training or loading the vector index and calculating probabilities.

        :param utterances: List of training utterances.
        :param labels: List of multi-label targets for each utterance.
        :raises TypeError: If the labels are not multi-label.
        :raises ValueError: If the vector index mismatches the provided utterances.
        """
        if not isinstance(labels[0], list):
            msg = "mlknn scorer support only multilabel input"
            raise TypeError(msg)

        self.n_classes = len(labels[0])

        vector_index_client = VectorIndexClient(self.device, self.db_dir)

        if self.prebuilt_index:
            # this happens only when LinearScorer is within Pipeline opimization after RetrievalNode optimization
            self.vector_index = vector_index_client.get_index(self.embedder_name)
            if len(utterances) != len(self.vector_index.texts):
                msg = "Vector index mismatches provided utterances"
                raise ValueError(msg)
        else:
            self.vector_index = vector_index_client.create_index(self.embedder_name, utterances, labels)

        self.features = (
            self.vector_index.embedder.embed(utterances)
            if self.vector_index.is_empty()
            else self.vector_index.get_all_embeddings()
        )
        self.labels = np.array(labels)
        self._prior_prob_true, self._prior_prob_false = self._compute_prior(self.labels)
        self._cond_prob_true, self._cond_prob_false = self._compute_cond()

    def _compute_prior(self, y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute prior probabilities for each class.

        :param y: Array of labels (multi-label format).
        :return: Tuple of prior probabilities for true and false labels.
        """
        prior_prob_true = (self.s + y.sum(axis=0)) / (self.s * 2 + y.shape[0])
        prior_prob_false = 1 - prior_prob_true
        return prior_prob_true, prior_prob_false

    def _compute_cond(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute conditional probabilities for neighbors.

        :return: Tuple of conditional probabilities for true and false labels.
        """
        c = np.zeros((self.n_classes, self.k + 1), dtype=int)
        cn = np.zeros((self.n_classes, self.k + 1), dtype=int)

        neighbors_labels, _ = self._get_neighbors(self.features)

        for i in range(self.labels.shape[0]):
            deltas = np.sum(neighbors_labels[i], axis=0).astype(int)
            idx_helper = np.arange(self.n_classes)
            deltas_idx = deltas[idx_helper]
            c[idx_helper, deltas_idx] += self.labels[i]
            cn[idx_helper, deltas_idx] += 1 - self.labels[i]

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_prob_true = (self.s + c) / (self.s * (self.k + 1) + c_sum[:, None])
        cond_prob_false = (self.s + cn) / (self.s * (self.k + 1) + cn_sum[:, None])

        return cond_prob_true, cond_prob_false

    def _get_neighbors(
        self,
        queries: list[str] | NDArray[Any],
    ) -> tuple[NDArray[np.int64], list[list[str]]]:
        labels, _, neighbors = self.vector_index.query(
            queries,
            self.k + self.ignore_first_neighbours,
        )
        return (
            np.array([candidates[self.ignore_first_neighbours :] for candidates in labels]),
            neighbors,
        )

    def predict_labels(self, utterances: list[str], thresh: float = 0.5) -> NDArray[np.int64]:
        """
        Predict labels for the given utterances.

        :param utterances: List of query utterances.
        :param thresh: Threshold for binary classification, defaults to 0.5.
        :return: Predicted labels as a binary array.
        """
        probas = self.predict(utterances)
        return (probas > thresh).astype(int)

    def predict(self, utterances: list[str]) -> NDArray[np.float64]:
        """
        Predict probabilities for the given utterances.

        :param utterances: List of query utterances.
        :return: Array of predicted probabilities for each class.
        """
        return self._predict(utterances)[0]

    def predict_with_metadata(self, utterances: list[str]) -> tuple[NDArray[Any], list[dict[str, Any]] | None]:
        """
        Predict probabilities along with metadata for the given utterances.

        :param utterances: List of query utterances.
        :return: Tuple of probabilities and metadata with neighbor information.
        """
        scores, neighbors = self._predict(utterances)
        metadata = [{"neighbors": utterance_neighbors} for utterance_neighbors in neighbors]
        return scores, metadata

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the vector index."""
        self.vector_index.clear_ram()

    def dump(self, path: str) -> None:
        """
        Save the MLKnnScorer's metadata and probabilities to disk.

        :param path: Path to the directory where assets will be dumped.
        """
        self.metadata = MLKnnScorerDumpMetadata(
            db_dir=self.db_dir,
            n_classes=self.n_classes,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)

        arrays_to_save = ArrayToSave(
            prior_prob_true=self._prior_prob_true,
            prior_prob_false=self._prior_prob_false,
            cond_prob_true=self._cond_prob_true,
            cond_prob_false=self._cond_prob_false,
        )
        np.savez(dump_dir / self.arrays_filename, **arrays_to_save)

    def load(self, path: str) -> None:
        """
        Load the MLKnnScorer's metadata and probabilities from disk.

        :param path: Path to the directory containing the dumped assets.
        """
        dump_dir = Path(path)

        with (dump_dir / self.metadata_dict_name).open() as file:
            self.metadata: MLKnnScorerDumpMetadata = json.load(file)
        self.n_classes = self.metadata["n_classes"]

        arrays: ArrayToSave = np.load(dump_dir / self.arrays_filename)

        self._prior_prob_true = arrays["prior_prob_true"]
        self._prior_prob_false = arrays["prior_prob_false"]
        self._cond_prob_true = arrays["cond_prob_true"]
        self._cond_prob_false = arrays["cond_prob_false"]

        vector_index_client = VectorIndexClient(
            device=self.device,
            db_dir=self.metadata["db_dir"],
            embedder_batch_size=self.metadata["batch_size"],
            embedder_max_length=self.metadata["max_length"],
        )
        self.vector_index = vector_index_client.get_index(self.embedder_name)

    def _predict(
        self,
        utterances: list[str],
    ) -> tuple[NDArray[np.float64], list[list[str]]]:
        result = np.zeros((len(utterances), self.n_classes), dtype=float)
        neighbors_labels, neighbors = self._get_neighbors(utterances)

        for instance in range(neighbors_labels.shape[0]):
            deltas = np.sum(neighbors_labels[instance], axis=0).astype(int)

            for label in range(self.n_classes):
                p_true = self._prior_prob_true[label] * self._cond_prob_true[label, deltas[label]]
                p_false = self._prior_prob_false[label] * self._cond_prob_false[label, deltas[label]]
                result[instance, label] = p_true / (p_true + p_false)

        return result, neighbors
