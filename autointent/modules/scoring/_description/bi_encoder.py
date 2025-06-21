"""DescriptionScorer classes for scoring utterances based on intent descriptions."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import PositiveFloat

from autointent import Context, Embedder
from autointent.configs import EmbedderConfig, TaskTypeEnum

from .base import BaseDescriptionScorer


class BiEncoderDescriptionScorer(BaseDescriptionScorer):
    """Bi-encoder description scorer that embeds utterances and descriptions separately.

    This scorer uses a bi-encoder architecture where both utterances and descriptions
    are embedded separately, then cosine similarity is computed between them.

    Args:
        embedder_config: Config of the embedder model
        temperature: Temperature parameter for scaling logits, defaults to 1.0
    """

    name = "description_bi"

    def __init__(
        self,
        embedder_config: EmbedderConfig | str | dict[str, Any] | None = None,
        temperature: PositiveFloat = 1.0,
    ) -> None:
        super().__init__(temperature)
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

        return cls(
            temperature=temperature,
            embedder_config=embedder_config,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        """Get implicit initialization parameters for this scorer."""
        return {"embedder_config": self.embedder_config.model_dump()}

    def _fit_implementation(self, utterances: list[str], descriptions: list[str]) -> None:
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
