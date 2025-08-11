"""RetrievalAimedEmbedding class for a proxy optimization of embedding."""

from typing import Any

from pydantic import PositiveInt

from autointent import Context, VectorIndex
from autointent.configs import EmbedderConfig
from autointent.context.optimization_info import EmbeddingArtifact
from autointent.custom_types import ListOfLabels
from autointent.metrics import RETRIEVAL_METRICS_MULTICLASS, RETRIEVAL_METRICS_MULTILABEL
from autointent.modules.base import BaseEmbedding


class RetrievalAimedEmbedding(BaseEmbedding):
    """Module for configuring embeddings optimized for retrieval tasks.

    The main purpose of this module is to be used at embedding node for optimizing
    embedding configuration using its retrieval quality as a sort of proxy metric.

    Args:
        k: Number of nearest neighbors to retrieve
        embedder_config: Config of the embedder used for creating embeddings

    Examples:
    --------

    .. testcode::

        from autointent.modules.embedding import RetrievalAimedEmbedding
        utterances = ["bye", "how are you?", "good morning"]
        labels = [0, 1, 1]
        retrieval = RetrievalAimedEmbedding(
            k=2,
            embedder_config="sergeyzh/rubert-tiny-turbo",
        )
        retrieval.fit(utterances, labels)

    """

    _vector_index: VectorIndex
    name = "retrieval"
    supports_multiclass = True
    supports_multilabel = True
    supports_oos = False

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        k: PositiveInt = 10,
    ) -> None:
        self.k = k
        embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self.embedder_config = embedder_config

        if self.k < 0 or not isinstance(self.k, int):
            msg = "`k` argument of `RetrievalAimedEmbedding` must be a positive int"
            raise ValueError(msg)

    @classmethod
    def from_context(
        cls,
        context: Context,
        embedder_config: EmbedderConfig | str | None = None,
        k: PositiveInt = 10,
    ) -> "RetrievalAimedEmbedding":
        """Create an instance using a Context object.

        Args:
            context: The context containing configurations and utilities
            k: Number of nearest neighbors to retrieve
            embedder_config: Config of the embedder to use
        """
        return cls(
            k=k,
            embedder_config=embedder_config,
        )

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        """Fit the vector index using the provided utterances and labels.

        Args:
            utterances: List of text data to index
            labels: List of corresponding labels for the utterances
        """
        self._validate_task(labels)

        self._vector_index = VectorIndex(
            self.embedder_config,
        )
        self._vector_index.add(utterances, labels)

    def score_ho(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """Evaluate the embedding model using specified metric functions.

        Args:
            context: Context containing test data and labels
            metrics: List of metric names to compute

        Returns:
            Dictionary of computed metric values for the test set
        """
        train_utterances, train_labels = self.get_train_data(context)
        self.fit(train_utterances, train_labels)

        val_utterances = context.data_handler.validation_utterances(0)
        val_labels = context.data_handler.validation_labels(0)
        predictions = self.predict(val_utterances)

        metrics_dict = RETRIEVAL_METRICS_MULTILABEL if context.is_multilabel() else RETRIEVAL_METRICS_MULTICLASS
        chosen_metrics = {name: fn for name, fn in metrics_dict.items() if name in metrics}
        return self.score_metrics_ho((val_labels, predictions), chosen_metrics)

    def score_cv(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """Evaluate the embedding model using specified metric functions.

        Args:
            context: Context containing test data and labels
            metrics: List of metric names to compute

        Returns:
            Dictionary of computed metric values for the test set
        """
        metrics_dict = RETRIEVAL_METRICS_MULTILABEL if context.is_multilabel() else RETRIEVAL_METRICS_MULTICLASS
        chosen_metrics = {name: fn for name, fn in metrics_dict.items() if name in metrics}

        metrics_calculated, _ = self.score_metrics_cv(chosen_metrics, context.data_handler.validation_iterator())
        return metrics_calculated

    def get_assets(self) -> EmbeddingArtifact:
        """Get the retriever artifacts for this module.

        Returns:
            A EmbeddingArtifact object containing embedder information
        """
        return EmbeddingArtifact(config=self.embedder_config)

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the vector index."""
        if hasattr(self, "_vector_index"):
            self._vector_index.clear_ram()

    def predict(self, utterances: list[str]) -> list[ListOfLabels]:
        """Predict the nearest neighbors for a list of utterances.

        Args:
            utterances: List of utterances for which nearest neighbors are to be retrieved

        Returns:
            List of labels for each retrieved utterance
        """
        predictions, _, _ = self._vector_index.query(utterances, self.k)
        return predictions
