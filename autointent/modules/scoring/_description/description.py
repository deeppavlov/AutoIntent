"""DescriptionScorer class for scoring utterances based on intent descriptions."""

from typing import Any

import numpy as np
import scipy
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity

from autointent import Context, Embedder
from autointent.context.optimization_info import ScorerArtifact
from autointent.custom_types import ListOfLabels
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL
from autointent.modules.abc import ScoringModule


class DescriptionScorer(ScoringModule):
    r"""
    Scoring module that scores utterances based on similarity to intent descriptions.

    DescriptionScorer embeds both the utterances and the intent descriptions, then computes a similarity score
    between the two, using either cosine similarity and softmax.

    :ivar embedder: The embedder used to generate embeddings for utterances and descriptions.
    :ivar name: Name of the scorer, defaults to "description".

    """

    _embedder: Embedder
    name = "description"
    _n_classes: int
    _multilabel: bool
    _description_vectors: NDArray[Any]
    supports_multiclass = True
    supports_multilabel = True

    def __init__(
        self,
        embedder_name: str,
        temperature: float = 1.0,
        embedder_device: str = "cpu",
        embedder_batch_size: int = 32,
        embedder_max_length: int | None = None,
        embedder_use_cache: bool = True,
    ) -> None:
        """
        Initialize the DescriptionScorer.

        :param embedder_name: Name of the embedder model.
        :param temperature: Temperature parameter for scaling logits, defaults to 1.0.
        :param embedder_device: Device to run the embedder on, e.g., "cpu" or "cuda".
        :param embedder_batch_size: Batch size for embedding generation, defaults to 32.
        :param embedder_max_length: Maximum sequence length for embedding, defaults to None.
        :param embedder_use_cache: Flag indicating whether to cache intermediate embeddings.
        """
        self.temperature = temperature
        self.embedder_device = embedder_device
        self.embedder_name = embedder_name
        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_length = embedder_max_length
        self.embedder_use_cache = embedder_use_cache

    @classmethod
    def from_context(
        cls,
        context: Context,
        temperature: float,
        embedder_name: str | None = None,
    ) -> "DescriptionScorer":
        """
        Create a DescriptionScorer instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param temperature: Temperature parameter for scaling logits.
        :param embedder_name: Name of the embedder model. If None, the best embedder is used.
        :return: Initialized DescriptionScorer instance.
        """
        if embedder_name is None:
            embedder_name = context.optimization_info.get_best_embedder()

        return cls(
            temperature=temperature,
            embedder_device=context.get_device(),
            embedder_name=embedder_name,
            embedder_use_cache=context.get_use_cache(),
            embedder_batch_size=context.get_batch_size(),
            embedder_max_length=context.get_max_length(),
        )

    def get_embedder_name(self) -> str:
        """
        Get the name of the embedder.

        :return: Embedder name.
        """
        return self.embedder_name

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
        descriptions: list[str],
    ) -> None:
        """
        Fit the scorer by embedding utterances and descriptions.

        :param utterances: List of utterances to embed.
        :param labels: List of labels corresponding to the utterances.
        :param descriptions: List of intent descriptions.
        :raises ValueError: If descriptions contain None values or embeddings mismatch utterances.
        """
        if hasattr(self, "_embedder"):
            self._embedder.clear_ram()

        self._validate_task(labels)

        if any(description is None for description in descriptions):
            error_text = (
                "Some intent descriptions (label_description) are missing (None). "
                "Please ensure all intents have descriptions."
            )
            raise ValueError(error_text)

        embedder = Embedder(
            device=self.embedder_device,
            model_name_or_path=self.embedder_name,
            batch_size=self.embedder_batch_size,
            max_length=self.embedder_max_length,
            use_cache=self.embedder_use_cache,
        )

        self._description_vectors = embedder.embed(descriptions)
        self._embedder = embedder

    def predict(self, utterances: list[str]) -> NDArray[np.float64]:
        """
        Predict scores for utterances based on similarity to intent descriptions.

        :param utterances: List of utterances to score.
        :return: Array of probabilities for each utterance.
        """
        utterance_vectors = self._embedder.embed(utterances)
        similarities: NDArray[np.float64] = cosine_similarity(utterance_vectors, self._description_vectors)

        if self._multilabel:
            probabilites = scipy.special.expit(similarities / self.temperature)
        else:
            probabilites = scipy.special.softmax(similarities / self.temperature, axis=1)
        return probabilites  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the embedder."""
        self._embedder.clear_ram()

    def get_train_data(self, context: Context) -> tuple[list[str], ListOfLabels, list[str]]:
        return (  # type: ignore[return-value]
            context.data_handler.train_utterances(0),
            context.data_handler.train_labels(0),
            context.data_handler.intent_descriptions,
        )

    def score_cv(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """
        Evaluate the scorer on a test set and compute the specified metric.

        :param context: Context containing test set and other data.
        :param split: Target split
        :return: Computed metrics value for the test set or error code of metrics
        """
        metrics_dict = SCORING_METRICS_MULTILABEL if context.is_multilabel() else SCORING_METRICS_MULTICLASS
        chosen_metrics = {name: fn for name, fn in metrics_dict.items() if name in metrics}

        metrics_calculated, all_val_scores = self.score_metrics_cv(
            chosen_metrics,
            context.data_handler.validation_iterator(),
            descriptions=context.data_handler.intent_descriptions,
        )

        self._artifact = ScorerArtifact(folded_scores=all_val_scores)

        return metrics_calculated
