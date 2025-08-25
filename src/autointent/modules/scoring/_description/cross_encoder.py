"""DescriptionScorer class for scoring utterances based on intent descriptions."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import PositiveFloat

from autointent import Context, Ranker
from autointent.configs import CrossEncoderConfig

from .base import BaseDescriptionScorer


class CrossEncoderDescriptionScorer(BaseDescriptionScorer):
    """Cross-encoder description scorer for zero-shot intent classification.

    This scorer uses a cross-encoder architecture that directly computes similarity
    scores between each utterance-description pair by passing them together through
    a transformer model. Unlike bi-encoders that embed texts separately, cross-encoders
    can capture more complex interactions between utterances and descriptions, often
    leading to higher accuracy at the cost of computational efficiency.

    This is a zero-shot approach that doesn't require training examples, only intent
    descriptions. The cross-encoder processes each utterance-description pair separately
    during inference, making it more computationally intensive but potentially more accurate.

    Args:
        cross_encoder_config: Configuration for the cross-encoder model (HuggingFace model name or config)
        temperature: Temperature parameter for scaling logits before softmax/sigmoid (default: 1.0)
        multilabel: Flag indicating classification task type

    Example:
    --------
    .. testcode::

        from autointent.modules.scoring import CrossEncoderDescriptionScorer

        # Initialize cross-encoder scorer
        scorer = CrossEncoderDescriptionScorer(
            cross_encoder_config="cross-encoder/ms-marco-MiniLM-L-6-v2",
            temperature=1.2
        )

        # Zero-shot classification with intent descriptions
        descriptions = [
            "User wants to book or reserve transportation like flights, trains, or hotels",
            "User wants to cancel an existing booking or reservation",
            "User asks about weather conditions or forecasts"
        ]

        # Fit using descriptions only (zero-shot approach)
        scorer.fit([], [], descriptions)

        # Make predictions on new utterances
        test_utterances = ["Reserve a hotel room", "Delete my booking"]
        probabilities = scorer.predict(test_utterances)
    """

    name = "description_cross"

    def __init__(
        self,
        cross_encoder_config: CrossEncoderConfig | str | dict[str, Any] | None = None,
        temperature: PositiveFloat = 1.0,
        multilabel: bool = False,
    ) -> None:
        super().__init__(temperature=temperature, multilabel=multilabel)
        self.cross_encoder_config = CrossEncoderConfig.from_search_config(cross_encoder_config)
        self._cross_encoder: Ranker | None = None
        self._description_texts: list[str] | None = None

    @classmethod
    def from_context(
        cls,
        context: Context,
        temperature: PositiveFloat = 1.0,
        cross_encoder_config: CrossEncoderConfig | str | None = None,
    ) -> "CrossEncoderDescriptionScorer":
        """Create a CrossEncoderDescriptionScorer instance using a Context object.

        Args:
            context: Context containing configurations and utilities
            temperature: Temperature parameter for scaling logits
            cross_encoder_config: Config of the cross-encoder model. If None, the default config is used

        Returns:
            Initialized CrossEncoderDescriptionScorer instance
        """
        if cross_encoder_config is None:
            cross_encoder_config = context.resolve_ranker()

        return cls(
            temperature=temperature, cross_encoder_config=cross_encoder_config, multilabel=context.is_multilabel()
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        """Get implicit initialization parameters for this scorer."""
        return {"cross_encoder_config": self.cross_encoder_config.model_dump(), "multilabel": self._multilabel}

    def _fit_implementation(self, descriptions: list[str]) -> None:
        """Fit the cross-encoder by storing descriptions.

        Args:
            utterances: List of utterances (not used in cross-encoder fitting)
            descriptions: List of intent descriptions to store
        """
        self._cross_encoder = Ranker(self.cross_encoder_config)
        self._description_texts = descriptions

    def _compute_similarities(self, utterances: list[str]) -> NDArray[np.float64]:
        """Compute similarities using cross-encoder approach.

        Args:
            utterances: List of utterances to score

        Returns:
            Array of similarity scores

        Raises:
            RuntimeError: If cross-encoder or description texts are not initialized
        """
        if self._cross_encoder is None:
            error_text = "Cross encoder is not initialized. Call fit() before predict()."
            raise RuntimeError(error_text)

        if self._description_texts is None:
            error_text = "Description texts are not initialized. Call fit() before predict()."
            raise RuntimeError(error_text)

        pairs = [(utterance, description) for utterance in utterances for description in self._description_texts]

        scores = self._cross_encoder.predict(pairs)
        return np.array(scores, dtype=np.float64).reshape(len(utterances), len(self._description_texts))

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the cross-encoder."""
        if self._cross_encoder is not None:
            self._cross_encoder.clear_ram()
