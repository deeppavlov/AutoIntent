"""RetrievalAimedEmbedding class for a proxy optimization of embedding."""

from typing import Any

from pydantic import PositiveInt

from autointent import Context, Embedder, VectorIndex
from autointent.configs import (
    EmbedderConfig,
    EmbedderFineTuningConfig,
    VectorIndexConfig,
    get_default_vector_index_config,
)
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
        ft_config: settings for fine-tuning embeddings

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
        vector_index_config: VectorIndexConfig | None = None,
        k: PositiveInt = 10,
        ft_config: EmbedderFineTuningConfig | dict[str, Any] | None = None,
    ) -> None:
        self.k = k
        self._embedder = Embedder(EmbedderConfig.from_search_config(embedder_config))
        self.vector_index_config = vector_index_config or get_default_vector_index_config()
        self.ft_config = EmbedderFineTuningConfig.from_search_config(ft_config)

        if self.k < 0 or not isinstance(self.k, int):
            msg = "`k` argument of `RetrievalAimedEmbedding` must be a positive int"
            raise ValueError(msg)

    @classmethod
    def from_context(
        cls,
        context: Context,
        embedder_config: EmbedderConfig | str | None = None,
        k: PositiveInt = 10,
        ft_config: EmbedderFineTuningConfig | dict[str, Any] | None = None,
    ) -> "RetrievalAimedEmbedding":
        """Create an instance using a Context object.

        Args:
            context: The context containing configurations and utilities
            k: Number of nearest neighbors to retrieve
            embedder_config: Config of the embedder to use
            ft_config: settings for fine-tuning embeddings
        """
        return cls(
            k=k,
            embedder_config=embedder_config,
            vector_index_config=context.vector_index_config,
            ft_config=ft_config,
        )

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        """Fit the vector index using the provided utterances and labels.

        Args:
            utterances: List of text data to index
            labels: List of corresponding labels for the utterances
        """
        self._validate_task(labels)

        if self.ft_config is not None:
            self._embedder.train(utterances=utterances, labels=labels, config=self.ft_config)

        self._vector_index = VectorIndex(self._embedder, config=self.vector_index_config)
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
        return EmbeddingArtifact(config=self._embedder.config)

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
        _, documents = self._vector_index.query(utterances, self.k)
        return [[n.label for n in neigs] for neigs in documents]  # type: ignore[misc]
