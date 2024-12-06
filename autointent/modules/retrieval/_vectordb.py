"""VectorDBModule class for managing and interacting with a vector database for retrieval tasks."""

import json
from pathlib import Path
from typing import Literal

from autointent.context import Context
from autointent.context.optimization_info import RetrieverArtifact
from autointent.context.vector_index_client import VectorIndex, VectorIndexClient, get_db_dir
from autointent.custom_types import BaseMetadataDict, LabelType
from autointent.metrics import RetrievalMetricFn
from autointent.modules.abc import RetrievalModule


class VectorDBMetadata(BaseMetadataDict):
    """Metadata class for VectorDBModule."""

    db_dir: str
    batch_size: int
    max_length: int | None


class VectorDBModule(RetrievalModule):
    r"""
    Module for managing retrieval operations using a vector database.

    VectorDBModule provides methods for indexing, querying, and managing a vector database for tasks
    such as nearest neighbor retrieval.

    :ivar vector_index: The vector index used for nearest neighbor retrieval.
    :ivar name: Name of the module, defaults to "vector_db".

    Examples
    --------
    Creating and fitting the VectorDBModule:
    >>> from your_module import VectorDBModule
    >>> utterances = ["hello world", "how are you?", "good morning"]
    >>> labels = [1, 2, 3]
    >>> vector_db = VectorDBModule(k=2, embedder_name="some_embedder", db_dir="./db", device="cpu")
    >>> vector_db.fit(utterances, labels)
    >>> def retrieval_metric_fn(true_labels, predicted_labels):
    >>>     # Custom metric function (e.g., accuracy or F1 score)
    >>>     return sum([1 if true == pred else 0 for true, pred \\
    >>>         in zip(true_labels, predicted_labels)]) / len(true_labels)
    >>> score = vector_db.score(context, retrieval_metric_fn)
    >>> print(score)

    Performing predictions:
    >>> predictions = vector_db.predict(["how is the weather today?"])
    >>> print(predictions)

    Saving and loading the model:
    >>> vector_db.dump("outputs/")
    >>> loaded_vector_db = VectorDBModule(k=2, embedder_name="some_embedder", db_dir="./db", device="cpu")
    >>> loaded_vector_db.load("outputs/")
    >>> print(loaded_vector_db.vector_index)
    """

    vector_index: VectorIndex
    name = "vector_db"

    def __init__(
        self,
        k: int,
        embedder_name: str,
        db_dir: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int | None = None,
        embedder_use_cache: bool = False,
    ) -> None:
        """
        Initialize the VectorDBModule.

        :param k: Number of nearest neighbors to retrieve.
        :param embedder_name: Name of the embedder used for creating embeddings.
        :param db_dir: Path to the database directory. If None, defaults will be used.
        :param device: Device to run operations on, e.g., "cpu" or "cuda".
        :param batch_size: Batch size for embedding generation.
        :param max_length: Maximum sequence length for embeddings. None if not set.
        :param embedder_use_cache: Flag indicating whether to cache intermediate embeddings.
        """
        self.embedder_name = embedder_name
        self.device = device
        self._db_dir = db_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedder_use_cache = embedder_use_cache

        super().__init__(k=k)

    @classmethod
    def from_context(
        cls,
        context: Context,
        k: int,
        embedder_name: str,
    ) -> "VectorDBModule":
        """
        Create a VectorDBModule instance using a Context object.

        :param context: The context containing configurations and utilities.
        :param k: Number of nearest neighbors to retrieve.
        :param embedder_name: Name of the embedder to use.
        :return: Initialized VectorDBModule instance.
        """
        return cls(
            k=k,
            embedder_name=embedder_name,
            db_dir=str(context.get_db_dir()),
            device=context.get_device(),
            batch_size=context.get_batch_size(),
            max_length=context.get_max_length(),
            embedder_use_cache=context.get_use_cache(),
        )

    @property
    def db_dir(self) -> str:
        """
        Get the directory for the vector database.

        :return: Path to the database directory.
        """
        if self._db_dir is None:
            self._db_dir = str(get_db_dir())
        return self._db_dir

    def fit(self, utterances: list[str], labels: list[LabelType]) -> None:
        """
        Fit the vector index using the provided utterances and labels.

        :param utterances: List of text data to index.
        :param labels: List of corresponding labels for the utterances.
        """
        vector_index_client = VectorIndexClient(
            self.device,
            self.db_dir,
            embedder_batch_size=self.batch_size,
            embedder_max_length=self.max_length,
            embedder_use_cache=self.embedder_use_cache,
        )
        self.vector_index = vector_index_client.create_index(self.embedder_name, utterances, labels)

    def score(
        self,
        context: Context,
        split: Literal["validation", "test"],
        metric_fn: RetrievalMetricFn,
    ) -> float:
        """
        Evaluate the retrieval model using a specified metric function.

        :param context: The context containing test data and labels.
        :param split: Target split
        :param metric_fn: Function to compute the retrieval metric.
        :return: Computed metric score.
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
        predictions, _, _ = self.vector_index.query(utterances, self.k)
        return metric_fn(labels, predictions)

    def get_assets(self) -> RetrieverArtifact:
        """
        Get the retriever artifacts for this module.

        :return: A RetrieverArtifact object containing embedder information.
        """
        return RetrieverArtifact(embedder_name=self.embedder_name)

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the vector index."""
        self.vector_index.clear_ram()

    def dump(self, path: str) -> None:
        """
        Save the module's metadata and vector index to a specified directory.

        :param path: Path to the directory where assets will be dumped.
        """
        self.metadata = VectorDBMetadata(
            batch_size=self.batch_size,
            max_length=self.max_length,
            db_dir=self.db_dir,
        )

        dump_dir = Path(path)
        with (dump_dir / self.metadata_dict_name).open("w") as file:
            json.dump(self.metadata, file, indent=4)
        self.vector_index.dump(dump_dir)

    def load(self, path: str) -> None:
        """
        Load the module's metadata and vector index from a specified directory.

        :param path: Path to the directory containing the dumped assets.
        """
        dump_dir = Path(path)
        with (dump_dir / self.metadata_dict_name).open() as file:
            self.metadata: VectorDBMetadata = json.load(file)

        vector_index_client = VectorIndexClient(
            device=self.device,
            db_dir=self.metadata["db_dir"],
            embedder_batch_size=self.metadata["batch_size"],
            embedder_max_length=self.metadata["max_length"],
            embedder_use_cache=self.embedder_use_cache,
        )
        self.vector_index = vector_index_client.get_index(self.embedder_name)

    def predict(self, utterances: list[str]) -> tuple[list[list[int | list[int]]], list[list[float]], list[list[str]]]:
        """
        Predict the nearest neighbors for a list of utterances.

        :param utterances: List of utterances for which nearest neighbors are to be retrieved.
        :return: A tuple containing:
            - labels: List of retrieved labels for each utterance.
            - distances: List of distances to the nearest neighbors.
            - texts: List of retrieved text data corresponding to the neighbors.
        """
        return self.vector_index.query(
            utterances,
            self.k,
        )
