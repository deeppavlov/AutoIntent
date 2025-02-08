"""DescriptionScorer class for scoring utterances based on intent descriptions."""

from typing import Any

import numpy as np
import scipy
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity

from autointent import Context, Embedder
from autointent.custom_types import ListOfLabels
from autointent.modules.abc import ScoringModule
from autointent.schemas._schemas import EmbedderConfig


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
        embedder_config: EmbedderConfig | str,
        temperature: float = 1.0,
    ) -> None:
        """
        Initialize the DescriptionScorer.

        :param embedder_config: Name of the embedder model.
        :param temperature: Temperature parameter for scaling logits, defaults to 1.0.
        """
        self.temperature = temperature
        if isinstance(embedder_config, dict):
            embedder_config = EmbedderConfig(**embedder_config)
        if isinstance(embedder_config, str):
            embedder_config = EmbedderConfig(model_name=embedder_config)

        self.embedder_config = embedder_config

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
        :param embedder_config: Name of the embedder model. If None, the best embedder is used.
        :return: Initialized DescriptionScorer instance.
        """
        if embedder_config is None:
            embedder_config = context.optimization_info.get_best_embedder()

        return cls(
            temperature=temperature,
            embedder_config=embedder_config,
        )

    def get_embedder_name(self) -> str:
        """
        Get the name of the embedder.

        :return: Embedder name.
        """
        return self.embedder_config

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
        self._validate_task(labels)

        if any(description is None for description in descriptions):
            error_text = (
                "Some intent descriptions (label_description) are missing (None). "
                "Please ensure all intents have descriptions."
            )
            raise ValueError(error_text)

        embedder = Embedder(
            self.embedder_config,
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
