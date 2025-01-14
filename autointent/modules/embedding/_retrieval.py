"""RetrievalEmbedding class for managing and interacting with a vector database for retrieval tasks."""

from pathlib import Path
from typing import Literal

from autointent import VectorIndex
from autointent.context import Context
from autointent.context.optimization_info import RetrieverArtifact
from autointent.custom_types import LabelType
from autointent.metrics import RETRIEVAL_METRICS_MULTICLASS, RETRIEVAL_METRICS_MULTILABEL
from autointent.modules.abc import EmbeddingModule


class RetrievalEmbedding(EmbeddingModule):
    r"""
    Module for managing retrieval operations using a vector database.

    RetrievalEmbedding provides methods for indexing, querying, and managing a vector database for tasks
    such as nearest neighbor retrieval.

    :ivar vector_index: The vector index used for nearest neighbor retrieval.
    :ivar name: Name of the module, defaults to "retrieval".

    Examples
    --------

    .. testcode::

        from autointent.modules.embedding import RetrievalEmbedding
        utterances = ["bye", "how are you?", "good morning"]
        labels = [0, 1, 1]
        retrieval = RetrievalEmbedding(
            k=2,
            embedder_name="sergeyzh/rubert-tiny-turbo",
        )
        retrieval.fit(utterances, labels)
        predictions = retrieval.predict(["how is the weather today?"])
        print(predictions)

    .. testoutput::

        ([[1, 1]], [[0.1525942087173462, 0.18616724014282227]], [['good morning', 'how are you?']])

    """

    _vector_index: VectorIndex
    name = "retrieval"

    def __init__(
        self,
        k: int,
        embedder_name: str,
        embedder_device: str = "cpu",
        embedder_batch_size: int = 32,
        embedder_max_length: int | None = None,
        embedder_use_cache: bool = True,
    ) -> None:
        """
        Initialize the RetrievalEmbedding.

        :param k: Number of nearest neighbors to retrieve.
        :param embedder_name: Name of the embedder used for creating embeddings.
        :param embedder_device: Device to run operations on, e.g., "cpu" or "cuda".
        :param batch_size: Batch size for embedding generation.
        :param max_length: Maximum sequence length for embeddings. None if not set.
        :param embedder_use_cache: Flag indicating whether to cache intermediate embeddings.
        """
        self.embedder_name = embedder_name
        self.embedder_device = embedder_device
        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_length = embedder_max_length
        self.embedder_use_cache = embedder_use_cache

        super().__init__(k=k)

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: int,
        embedder_name: str,
    ) -> "RetrievalEmbedding":
        """
        Create a RetrievalEmbedding instance using a Context object.

        :param context: The context containing configurations and utilities.
        :param k: Number of nearest neighbors to retrieve.
        :param embedder_name: Name of the embedder to use.
        :return: Initialized RetrievalEmbedding instance.
        """
        return cls(
            k=k,
            embedder_name=embedder_name,
            embedder_device=context.get_device(),
            embedder_batch_size=context.get_batch_size(),
            embedder_max_length=context.get_max_length(),
            embedder_use_cache=context.get_use_cache(),
        )

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        """
        Fit the vector index using the provided utterances and labels.

        :param utterances: List of text data to index.
        :param labels: List of corresponding labels for the utterances.
        """
        self._vector_index = VectorIndex(
            self.embedder_name,
            self.embedder_device,
            self.embedder_batch_size,
            self.embedder_max_length,
            self.embedder_use_cache,
        )
        self._vector_index.add(utterances, labels)

    def score(
        self,
        context: Context,
        split: Literal["validation", "test"],
    ) -> dict[str, float | str]:
        """
        Evaluate the embedding model using a specified metric function.

        :param context: The context containing test data and labels.
        :param split: Target split
        :return: Computed metrics value for the test set or error code of metrics
        """
        if split == "validation":
            utterances = context.data_handler.validation_utterances(0)
            labels = context.data_handler.validation_labels(0)
        elif split == "test":
            utterances = context.data_handler.test_utterances()
            labels = context.data_handler.test_labels()
        else:
            message = f"Invalid split '{split}' provided. Expected one of 'validation', or 'test'."
            raise ValueError(message)
        predictions, _, _ = self._vector_index.query(utterances, self.k)

        metrics_dict = RETRIEVAL_METRICS_MULTILABEL if context.is_multilabel() else RETRIEVAL_METRICS_MULTICLASS
        return self.score_metrics((labels, predictions), metrics_dict)

    def get_assets(self) -> RetrieverArtifact:
        """
        Get the retriever artifacts for this module.

        :return: A RetrieverArtifact object containing embedder information.
        """
        return RetrieverArtifact(embedder_name=self.embedder_name)

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the vector index."""
        self._vector_index.clear_ram()

    def dump(self, path: str) -> None:
        """
        Save the module's metadata and vector index to a specified directory.

        :param path: Path to the directory where assets will be dumped.
        """
        self._vector_index.dump(Path(path))

    def load(self, path: str) -> None:
        """
        Load the module's metadata and vector index from a specified directory.

        :param path: Path to the directory containing the dumped assets.
        """
        self._vector_index = VectorIndex.load(Path(path))

    def predict(self, utterances: list[str]) -> tuple[list[list[int | list[int]]], list[list[float]], list[list[str]]]:
        """
        Predict the nearest neighbors for a list of utterances.

        :param utterances: List of utterances for which nearest neighbors are to be retrieved.
        :return: A tuple containing:
            - labels: List of retrieved labels for each utterance.
            - distances: List of distances to the nearest neighbors.
            - texts: List of retrieved text data corresponding to the neighbors.
        """
        return self._vector_index.query(
            utterances,
            self.k,
        )
