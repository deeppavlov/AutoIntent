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
from autointent.schemas import EmbedderConfig, TaskTypeEnum


class DescriptionScorer(ScoringModule):
    r"""
    Scoring module that scores utterances based on similarity to intent descriptions.

    DescriptionScorer embeds both the utterances and the intent descriptions, then computes a similarity score
    between the two, using either cosine similarity and softmax.

    :ivar _embedder: The embedder used to generate embeddings for utterances and descriptions.
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
        embedder_config: EmbedderConfig | str | dict[str, Any],
        temperature: float = 1.0,
    ) -> None:
        """
        Initialize the DescriptionScorer.

        :param embedder_config: Config of the embedder model.
        :param temperature: Temperature parameter for scaling logits, defaults to 1.0.
        """
        self.temperature = temperature
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)

    @classmethod
    def from_context(
        cls,
        context: Context,
        temperature: float,
        embedder_config: EmbedderConfig | str | None = None,
    ) -> "DescriptionScorer":
        """
        Create a DescriptionScorer instance using a Context object.

        :param context: Context containing configurations and utilities.
        :param temperature: Temperature parameter for scaling logits.
        :param embedder_config: Config of the embedder model. If None, the best embedder is used.
        :return: Initialized DescriptionScorer instance.
        """
        if embedder_config is None:
            embedder_config = context.optimization_info.get_best_embedder()

        return cls(
            temperature=temperature,
            embedder_config=embedder_config,
        )

    def get_embedder_config(self) -> dict[str, Any]:
        """
        Get the name of the embedder.

        :return: Embedder name.
        """
        return self.embedder_config.model_dump()

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

        embedder = Embedder(self.embedder_config)

        self._description_vectors = embedder.embed(descriptions, TaskTypeEnum.sts)
        self._embedder = embedder

    def predict(self, utterances: list[str]) -> NDArray[np.float64]:
        """
        Predict scores for utterances based on similarity to intent descriptions.

        :param utterances: List of utterances to score.
        :return: Array of probabilities for each utterance.
        """
        utterance_vectors = self._embedder.embed(utterances, TaskTypeEnum.sts)
        similarities: NDArray[np.float64] = cosine_similarity(utterance_vectors, self._description_vectors)

        if self._multilabel:
            probabilities = scipy.special.expit(similarities / self.temperature)
        else:
            probabilities = scipy.special.softmax(similarities / self.temperature, axis=1)
        return probabilities  # type: ignore[no-any-return]

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
        :param metrics: List of metric names to compute.
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
