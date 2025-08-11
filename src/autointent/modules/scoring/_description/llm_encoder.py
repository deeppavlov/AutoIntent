"""LLMDescriptionScorer class for scoring utterances based on intent descriptions using LLM."""

import asyncio
import logging
from functools import partial
from pathlib import Path
from textwrap import dedent
from typing import Any

import aiometer
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from typing_extensions import assert_never

from autointent import Context
from autointent._dump_tools import Dumper
from autointent.configs._transformers import CrossEncoderConfig, EmbedderConfig
from autointent.generation import Generator, RetriesExceededError
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


class LLMDescriptionScorer(BaseDescriptionScorer):
    """LLM-based description scorer for zero-shot intent classification using structured output.

    This scorer uses a Large Language Model (LLM) with structured output to perform
    zero-shot intent classification. The LLM is prompted to categorize intent descriptions
    into three probability levels for each utterance:
    - Most probable (probability 1.0): Intents that are most likely to match the utterance
    - Promising (probability 0.5): Intents that are plausible but less confident
    - Unlikely (probability 0.0): All other intents (implicit)

    This approach leverages the reasoning capabilities of LLMs to understand complex
    relationships between utterances and intent descriptions, potentially achieving
    high accuracy for nuanced classification tasks. However, it requires API access
    to LLM services and can be slower and more expensive than encoder-based methods.

    Args:
        generator_config: Configuration for the Generator instance (LLM model settings)
        temperature: Temperature parameter for scaling classifier logits (default: 1.0)
        max_concurrent: Maximum number of concurrent async calls to LLM (default: 15)
        max_per_second: Maximum number of API calls per second for rate limiting (default: 10)
        max_retries: Maximum number of retry attempts for failed API calls (default: 3)
        multilabel: Flag indicating classification task type

    Example:
    --------
    .. code-block::

        from autointent.modules.scoring import LLMDescriptionScorer

        # Initialize LLM scorer with OpenAI GPT
        scorer = LLMDescriptionScorer(
            temperature=1.0,
            max_concurrent=10,
            max_per_second=5,
            max_retries=2
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

    name = "description_llm"

    def __init__(
        self,
        generator_config: dict[str, Any] | None = None,
        temperature: PositiveFloat = 1.0,
        max_concurrent: PositiveInt | None = 15,
        max_per_second: PositiveInt = 10,
        max_retries: PositiveInt = 3,
        multilabel: bool = False,
    ) -> None:
        super().__init__(temperature=temperature, multilabel=multilabel)

        self.generator_config = generator_config or {}
        self.max_concurrent = max_concurrent
        self.max_per_second = max_per_second
        self.max_retries = max_retries

    @classmethod
    def from_context(
        cls,
        context: Context,
        temperature: PositiveFloat = 1.0,
        generator_config: dict[str, Any] | None = None,
        max_concurrent: PositiveInt | None = 15,
        max_per_second: PositiveInt = 10,
        max_retries: PositiveInt = 3,
    ) -> "LLMDescriptionScorer":
        return cls(
            temperature=temperature,
            generator_config=generator_config,
            max_concurrent=max_concurrent,
            max_per_second=max_per_second,
            max_retries=max_retries,
            multilabel=context.is_multilabel(),
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {"multilabel": self._multilabel}

    def _fit_implementation(self, descriptions: list[str]) -> None:
        """Fit the LLM scorer by initializing the generator and storing descriptions.

        Args:
            utterances: List of utterances (not used in LLM fitting)
            descriptions: List of intent descriptions to store
        """
        self._generator = Generator(**self.generator_config)
        self._description_texts = descriptions
        self._init_event_loop()

    def _create_prompt(self, utterance: str, descriptions: list[str]) -> list[Message]:
        """Create a prompt for the LLM to categorize intent descriptions.

        Args:
            utterance: The text sample to categorize
            descriptions: List of intent descriptions to categorize

        Returns:
            List of messages for the LLM
        """
        descriptions_text = "\n".join(
            f"<description_{i+1}>\n{desc}\n</description_{i+1}>" for i, desc in enumerate(descriptions)
        )

        content = dedent(
            f"""You are an expert at categorizing text samples into intent categories.

            Given a text sample and a list of possible intent descriptions,\
            categorize each intent description into one of two categories:

            1. **Most Probable**: Intent descriptions that are most likely to correspond to the text sample
            2. **Promising**: Intent descriptions that are promising but you're not fully confident about

            <text_sample>
            {utterance}
            </text_sample>

            <possible_intent_descriptions>
            {descriptions_text}
            </possible_intent_descriptions>

            <instructions>
            Please categorize each intent description into the appropriate category\
            based on how well it matches the text sample.

            IMPORTANT:
            - Use the numbers (1, 2, 3, etc.) that correspond to each description's position in the list above.
            - You can skip putting intents to an "unlikely" category.\
            If an intent is not explicitly categorized as "most_probable" or "promising",\
            it is automatically assumed to be unlikely.
            - Only include intents in the categories if you have confidence in their classification.
            </instructions>
            """
        )

        return [Message(role=Role.USER, content=content)]

    def _process_utterance_sync(self, utterance: str) -> IntentCategorization | RetriesExceededError:
        try:
            messages = self._create_prompt(utterance, self._description_texts)

            return self._generator.get_structured_output_sync(
                messages=messages,
                output_model=IntentCategorization,
                max_retries=self.max_retries,
            )
        except RetriesExceededError as e:
            return e

    async def _process_utterance_async(self, utterance: str) -> IntentCategorization | RetriesExceededError:
        try:
            messages = self._create_prompt(utterance, self._description_texts)

            return await self._generator.get_structured_output_async(
                messages=messages,
                output_model=IntentCategorization,
                max_retries=self.max_retries,
            )
        except RetriesExceededError as e:
            return e

    def _compute_similarities(self, utterances: list[str]) -> NDArray[np.float64]:
        """Compute similarities using LLM categorization approach.

        Args:
            utterances: List of utterances to score

        Returns:
            Array of similarity scores

        Raises:
            RuntimeError: If generator or description texts are not initialized
        """
        if not (hasattr(self, "_generator") and hasattr(self, "_description_texts")):
            error_text = "Scorer is not initialized. Call fit() before predict()."
            raise RuntimeError(error_text)

        similarities = np.zeros((len(utterances), len(self._description_texts)), dtype=np.float64)

        if self.max_concurrent is None:
            categorizations = map(self._process_utterance_sync, utterances)
        else:
            task = aiometer.run_all(
                [partial(self._process_utterance_async, utt) for utt in utterances],
                max_at_once=self.max_concurrent,
                max_per_second=self.max_per_second,
            )
            categorizations = self._event_loop.run_until_complete(task)  # type: ignore[arg-type]

        for i, categorization in enumerate(categorizations):
            if isinstance(categorization, IntentCategorization):
                for j in range(len(self._description_texts)):
                    # Convert 1-based indices to 0-based
                    if (j + 1) in categorization.most_probable:
                        similarities[i, j] = 1.0
                    elif (j + 1) in categorization.promising:
                        similarities[i, j] = 0.5
                    else:
                        similarities[i, j] = 0.0

            elif isinstance(categorization, RetriesExceededError):
                similarities[i, :] = 1.0 / len(self._description_texts)
                msg = f"LLM categorization failed for utterance '{utterances[i]}'"
                logger.warning(msg)
            else:
                assert_never(categorization)

        return similarities

    def clear_cache(self) -> None:
        """Clear cached data in memory used by the generator."""
        # Generator doesn't have a clear_ram method, so we just set it to None
        if hasattr(self, "_generator"):
            delattr(self, "_generator")
        if hasattr(self, "_event_loop"):
            delattr(self, "_event_loop")

    def _init_event_loop(self) -> None:
        if self.max_concurrent is not None:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
            self._event_loop = loop

    def dump(self, path: str) -> None:
        Dumper.dump(self, Path(path), exclude=[asyncio.BaseEventLoop])

    @classmethod
    def load(
        cls,
        path: str,
        embedder_config: EmbedderConfig | None = None,
        cross_encoder_config: CrossEncoderConfig | None = None,
    ) -> "LLMDescriptionScorer":
        instance = super().load(path=path, embedder_config=embedder_config, cross_encoder_config=cross_encoder_config)
        instance._init_event_loop()  # noqa: SLF001
        return instance
