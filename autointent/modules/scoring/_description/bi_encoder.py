"""DescriptionScorer classes for scoring utterances based on intent descriptions."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import PositiveFloat

from autointent import Context, Embedder
from autointent.configs import EmbedderConfig, TaskTypeEnum

from .base import BaseDescriptionScorer


class BiEncoderDescriptionScorer(BaseDescriptionScorer):
    """Bi-encoder description scorer for zero-shot intent classification.

    This scorer uses a bi-encoder architecture where utterances and intent descriptions
    are embedded separately using the same encoder model, then cosine similarity is
    computed between utterance embeddings and description embeddings. This is a
    zero-shot approach that doesn't require training examples, only intent descriptions.

    The bi-encoder approach is efficient for inference as descriptions are embedded
    once during fitting, and only utterances need to be embedded during prediction.

    Args:
        embedder_config: Configuration for the embedder model (HuggingFace model name or config)
        temperature: Temperature parameter for scaling logits before softmax/sigmoid (default: 1.0)
        multilabel: Flag indicating classification task type

    Example:
    --------
    .. testcode::

        from autointent.modules.scoring import BiEncoderDescriptionScorer

        # Initialize bi-encoder scorer
        scorer = BiEncoderDescriptionScorer(
            embedder_config="sentence-transformers/all-MiniLM-L6-v2",
            temperature=0.8
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

    name = "description_bi"

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        temperature: PositiveFloat = 1.0,
        multilabel: bool = False,
    ) -> None:
        super().__init__(temperature=temperature, multilabel=multilabel)
        self.embedder_config = EmbedderConfig.from_search_config(embedder_config)
        self._embedder: Embedder | None = None
        self._description_vectors: NDArray[Any] | None = None

    @classmethod
    def from_context(
        cls,
        context: Context,
        temperature: PositiveFloat = 1.0,
        embedder_config: EmbedderConfig | str | None = None,
    ) -> "BiEncoderDescriptionScorer":
        """Create a BiEncoderDescriptionScorer instance using a Context object.

        Args:
            context: Context containing configurations and utilities
            temperature: Temperature parameter for scaling logits
            embedder_config: Config of the embedder model. If None, the best embedder is used

        Returns:
            Initialized BiEncoderDescriptionScorer instance
        """
        if embedder_config is None:
            embedder_config = context.resolve_embedder()

        return cls(temperature=temperature, embedder_config=embedder_config, multilabel=context.is_multilabel())

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        """Get implicit initialization parameters for this scorer."""
        return {"embedder_config": self.embedder_config.model_dump(), "multilabel": self._multilabel}

    def _fit_implementation(self, descriptions: list[str]) -> None:
        """Fit the bi-encoder by embedding descriptions.

        Args:
            utterances: List of utterances (not used in bi-encoder fitting)
            descriptions: List of intent descriptions to embed
        """
        embedder = Embedder(self.embedder_config)
        self._description_vectors = embedder.embed(descriptions, TaskTypeEnum.sts)
        self._embedder = embedder

    def _compute_similarities(self, utterances: list[str]) -> NDArray[np.float64]:
        """Compute similarities using bi-encoder approach.

        Args:
            utterances: List of utterances to score

        Returns:
            Array of similarity scores

        Raises:
            RuntimeError: If embedder or description vectors are not initialized
        """
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
        return similarities

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the embedder."""
        if self._embedder is not None:
            self._embedder.clear_ram()
