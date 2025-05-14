"""DescriptionScorer class for scoring utterances based on intent descriptions."""

from typing import Any, Literal

import numpy as np
import scipy
from numpy.typing import NDArray
from pydantic import PositiveFloat

from autointent import Context, Embedder, Ranker
from autointent.configs import CrossEncoderConfig, EmbedderConfig, TaskTypeEnum
from autointent.context.optimization_info import ScorerArtifact
from autointent.custom_types import ListOfLabels
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL
from autointent.modules.base import BaseScorer


class DescriptionScorer(BaseScorer):
    """Scoring module that scores utterances based on similarity to intent descriptions.

    DescriptionScorer can use either a bi-encoder or cross-encoder architecture:
    - Bi-encoder: Embeds both utterances and descriptions separately, then computes cosine similarity
    - Cross-encoder: Directly computes similarity between each utterance-description pair

    Args:
        embedder_config: Config of the embedder model (for bi-encoder mode)
        cross_encoder_config: Config of the cross-encoder model (for cross-encoder mode)
        encoder_type: Type of encoder to use, either "bi" or "cross"
        temperature: Temperature parameter for scaling logits, defaults to 1.0
    """

    _embedder: Embedder | None = None
    _cross_encoder: Ranker | None = None
    name = "description"
    _n_classes: int
    _multilabel: bool
    _description_vectors: NDArray[Any] | None = None
    _description_texts: list[str] | None = None
    supports_multiclass = True
    supports_multilabel = True

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        cross_encoder_config: CrossEncoderConfig | str | dict[str, Any] | None = None,
        encoder_type: Literal["bi", "cross"] = "bi",
        temperature: PositiveFloat = 1.0,
    ) -> None:
        self.temperature = temperature
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self.cross_encoder_config = CrossEncoderConfig.from_search_config(cross_encoder_config)
        self._encoder_type = encoder_type

        if self.temperature < 0 or not isinstance(self.temperature, float | int):
            msg = "`temperature` argument of `DescriptionScorer` must be a positive float"
            raise ValueError(msg)

    @classmethod
    def from_context(
        cls,
        context: Context,
        temperature: PositiveFloat = 1.0,
        embedder_config: EmbedderConfig | str | None = None,
        cross_encoder_config: CrossEncoderConfig | str | None = None,
        encoder_type: Literal["bi", "cross"] = "bi",
    ) -> "DescriptionScorer":
        """Create a DescriptionScorer instance using a Context object.

        Args:
            context: Context containing configurations and utilities
            temperature: Temperature parameter for scaling logits
            embedder_config: Config of the embedder model. If None, the best embedder is used
            cross_encoder_config: Config of the cross-encoder model. If None, the default config is used
            encoder_type: Type of encoder to use, either "bi" or "cross"

        Returns:
            Initialized DescriptionScorer instance
        """
        if embedder_config is None and encoder_type == "bi":
            embedder_config = context.resolve_embedder()
        if cross_encoder_config is None and encoder_type == "cross":
            cross_encoder_config = context.resolve_ranker()

        return cls(
            temperature=temperature,
            embedder_config=embedder_config,
            cross_encoder_config=cross_encoder_config,
            encoder_type=encoder_type,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        res = {}
        if self._encoder_type == "bi":
            res["embedder_config"] = self.embedder_config.model_dump()
        else:
            res["cross_encoder_config"] = self.cross_encoder_config.model_dump()
        return res

    def fit(
        self,
        utterances: list[str],
        labels: ListOfLabels,
        descriptions: list[str],
    ) -> None:
        """Fit the scorer by embedding utterances and descriptions.

        Args:
            utterances: List of utterances to embed
            labels: List of labels corresponding to the utterances
            descriptions: List of intent descriptions

        Raises:
            ValueError: If descriptions contain None values or embeddings mismatch utterances
        """
        if hasattr(self, "_embedder") and self._embedder is not None:
            self._embedder.clear_ram()
        if hasattr(self, "_cross_encoder") and self._cross_encoder is not None:
            self._cross_encoder.clear_ram()

        self._validate_task(labels)

        if any(description is None for description in descriptions):
            error_text = (
                "Some intent descriptions (label_description) are missing (None). "
                "Please ensure all intents have descriptions."
            )
            raise ValueError(error_text)

        if self._encoder_type == "bi":
            embedder = Embedder(self.embedder_config)
            self._description_vectors = embedder.embed(descriptions, TaskTypeEnum.sts)
            self._embedder = embedder
            self._cross_encoder = None
            self._description_texts = None
        else:
            self._cross_encoder = Ranker(self.cross_encoder_config)
            self._description_texts = descriptions
            self._embedder = None
            self._description_vectors = None

    def predict(self, utterances: list[str]) -> NDArray[np.float64]:
        """Predict scores for utterances based on similarity to intent descriptions.

        Args:
            utterances: List of utterances to score

        Returns:
            Array of probabilities for each utterance
        """
        if self._encoder_type == "bi":
            if self._description_vectors is None:
                error_text = "Description vectors are not initialized. Call fit() before predict()."
                raise RuntimeError(error_text)

            if self._embedder is None:
                error_text = "Embedder is not initialized. Call fit() before predict()."
                raise RuntimeError(error_text)

            utterance_vectors = self._embedder.embed(utterances, TaskTypeEnum.sts)
            similarities: NDArray[np.float64] = np.array(
                self._embedder.similarity(utterance_vectors, self._description_vectors), dtype=np.float64
            )
        else:
            if self._cross_encoder is None:
                error_text = "Cross encoder is not initialized. Call fit() before predict()."
                raise RuntimeError(error_text)

            if self._description_texts is None:
                error_text = "Description texts are not initialized. Call fit() before predict()."
                raise RuntimeError(error_text)

            pairs = [(utterance, description) for utterance in utterances for description in self._description_texts]

            scores = self._cross_encoder.predict(pairs)
            similarities = np.array(scores, dtype=np.float64).reshape(len(utterances), len(self._description_texts))

        if self._multilabel:
            probabilities = scipy.special.expit(similarities / self.temperature)
        else:
            probabilities = scipy.special.softmax(similarities / self.temperature, axis=1)
        return probabilities  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the embedder or cross-encoder."""
        if self._embedder is not None:
            self._embedder.clear_ram()
        if self._cross_encoder is not None:
            self._cross_encoder.clear_ram()

    def get_train_data(self, context: Context) -> tuple[list[str], ListOfLabels, list[str]]:
        """Get training data from context.

        Args:
            context: Context containing training data

        Returns:
            Tuple containing utterances, labels, and descriptions
        """
        return (  # type: ignore[return-value]
            context.data_handler.train_utterances(0),
            context.data_handler.train_labels(0),
            context.data_handler.intent_descriptions,
        )

    def score_cv(self, context: Context, metrics: list[str]) -> dict[str, float]:
        """Evaluate the scorer on a test set and compute the specified metrics.

        Args:
            context: Context containing test set and other data
            metrics: List of metric names to compute

        Returns:
            Dictionary of computed metric values
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
