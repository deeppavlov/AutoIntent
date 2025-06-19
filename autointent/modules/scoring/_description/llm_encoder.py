"""LLMDescriptionScorer class for scoring utterances based on intent descriptions using LLM."""

import logging
from textwrap import dedent
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, PositiveFloat

from autointent import Context
from autointent.generation import Generator
from autointent.generation.chat_templates import Message, Role

from .base import BaseDescriptionScorer

logger = logging.getLogger(__name__)


class IntentCategorization(BaseModel):
    """Pydantic model for LLM output categorizing intents into three probability levels."""

    reasoning: str = Field(description="Preliminary planning and speculations on how to categorize given text.")
    most_probable: list[int] = Field(
        description=(
            "List of indices (1-based) of intent descriptions that are most "
            "likely to correspond to the text sample (probability 1.0)"
        )
    )
    promising: list[int] = Field(
        description=(
            "List of indices (1-based) of intent descriptions that are promising but not confident (probability 0.5)"
        )
    )
    unlikely: list[int] = Field(
        description=(
            "List of indices (1-based) of intent descriptions that are not sufficiently probable (probability 0.0)"
        )
    )


class LLMDescriptionScorer(BaseDescriptionScorer):
    """LLM-based description scorer that uses structured output to categorize intents.

    This scorer uses a language model with structured output to categorize intent descriptions
    into three categories based on their probability to correspond to a given text sample:
    - Most probable (probability 1.0)
    - Promising but not confident (probability 0.5)
    - Unlikely (probability 0.0)

    Args:
        generator_config: Configuration for the Generator instance
        temperature: Temperature parameter for scaling logits, defaults to 1.0
    """

    name = "description_llm"

    def __init__(
        self,
        generator_config: dict[str, Any] | None = None,
        temperature: PositiveFloat = 1.0,
    ) -> None:
        super().__init__(temperature)
        self.generator_config = generator_config or {}
        self._generator: Generator | None = None
        self._description_texts: list[str] | None = None

    @classmethod
    def from_context(
        cls,
        context: Context,
        temperature: PositiveFloat = 1.0,
        generator_config: dict[str, Any] | None = None,
    ) -> "LLMDescriptionScorer":
        """Create a LLMDescriptionScorer instance using a Context object.

        Args:
            context: Context containing configurations and utilities
            temperature: Temperature parameter for scaling logits
            generator_config: Configuration for the Generator instance

        Returns:
            Initialized LLMDescriptionScorer instance
        """
        return cls(
            temperature=temperature,
            generator_config=generator_config,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        """Get implicit initialization parameters for this scorer."""
        return {"generator_config": self.generator_config}

    def _fit_implementation(self, utterances: list[str], descriptions: list[str]) -> None:
        """Fit the LLM scorer by initializing the generator and storing descriptions.

        Args:
            utterances: List of utterances (not used in LLM fitting)
            descriptions: List of intent descriptions to store
        """
        self._generator = Generator(**self.generator_config)
        self._description_texts = descriptions

    def _create_prompt(self, utterance: str, descriptions: list[str]) -> list[Message]:
        """Create a prompt for the LLM to categorize intent descriptions.

        Args:
            utterance: The text sample to categorize
            descriptions: List of intent descriptions to categorize

        Returns:
            List of messages for the LLM
        """
        descriptions_text = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(descriptions))

        content = dedent(
            f"""You are an expert at categorizing text samples into intent categories.

            Given a text sample and a list of possible intent descriptions,\
            categorize each intent description into one of three categories:

            1. **Most Probable**: Intent descriptions that are most likely to correspond to the text sample
            2. **Promising**: Intent descriptions that are promising but you're not fully confident about
            3. **Unlikely**: Intent descriptions that are not sufficiently probable to correspond to the text sample

            Text Sample: "{utterance}"

            Possible Intent Descriptions:
            {descriptions_text}

            Please categorize each intent description into the appropriate category\
            based on how well it matches the text sample.

            IMPORTANT: Use the numbers (1, 2, 3, etc.) that correspond to each description's position in the list above.
            """
        )

        return [Message(role=Role.USER, content=content)]

    def _compute_similarities(self, utterances: list[str]) -> NDArray[np.float64]:
        """Compute similarities using LLM categorization approach.

        Args:
            utterances: List of utterances to score

        Returns:
            Array of similarity scores

        Raises:
            RuntimeError: If generator or description texts are not initialized
        """
        if self._generator is None:
            error_text = "Generator is not initialized. Call fit() before predict()."
            raise RuntimeError(error_text)

        if self._description_texts is None:
            error_text = "Description texts are not initialized. Call fit() before predict()."
            raise RuntimeError(error_text)

        similarities = np.zeros((len(utterances), len(self._description_texts)), dtype=np.float64)

        for i, utterance in enumerate(utterances):
            try:
                # Create prompt for this utterance
                messages = self._create_prompt(utterance, self._description_texts)

                # Get structured output from LLM
                categorization = self._generator.get_structured_output_sync(
                    messages=messages,
                    output_model=IntentCategorization,
                    backend="openai",
                    max_retries=3,
                )

                # Assign probabilities based on categorization using indices
                for j in range(len(self._description_texts)):
                    # Convert 1-based indices to 0-based
                    if (j + 1) in categorization.most_probable:
                        similarities[i, j] = 1.0
                    elif (j + 1) in categorization.promising:
                        similarities[i, j] = 0.5
                    else:
                        similarities[i, j] = 0.0

            except Exception as e:  # noqa: BLE001, PERF203
                # If LLM fails, assign uniform probabilities as fallback
                similarities[i, :] = 1.0 / len(self._description_texts)
                msg = f"LLM categorization failed for utterance '{utterance}': {e}"
                logger.warning(msg)

        return similarities

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the generator."""
        # Generator doesn't have a clear_ram method, so we just set it to None
        self._generator = None
