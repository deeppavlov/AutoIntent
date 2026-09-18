"""TypeSafeDescriptionScorer: zero-shot intent scoring with TypeSafe System One models (jev)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import aiometer
import numpy as np
import scipy
from dotenv import load_dotenv
from pydantic import BaseModel, PositiveFloat, PositiveInt

from autointent import Context
from autointent._deps import require
from autointent._hash import Hasher
from autointent.generation._cache import PydanticDiskCache

from .base import BaseDescriptionScorer

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

load_dotenv()

CACHE_SUBDIR = "typesafe_answers"
CHOICE_KEY = "intent"
DEFAULT_MODEL = "jev-latest"
DEFAULT_MODEL_ENV = "TYPESAFE_DEFAULT_MODEL"
_EPS = 1e-6

QuestionType = Literal["choice", "noul"]
_Result = tuple[list[float], int, int] | Exception
"""Either ``(probability_row, input_tokens, output_tokens)`` or the failure that produced no row."""


class TypeSafeAnswer(BaseModel):
    """Per-utterance probabilities in intent-description order, as stored in the disk cache."""

    probabilities: list[float]


def build_questions(question_type: QuestionType, descriptions: list[str]) -> dict[str, dict[str, Any]]:
    """Build the System One questions for one utterance as plain dicts (the SDK accepts them).

    ``choice`` asks a single question whose options are the descriptions; ``noul`` asks one
    yes/no question per description. Option keys are ``intent_{i}`` in description order.
    """
    if question_type == "choice":
        return {
            CHOICE_KEY: {
                "type": "choice",
                "instructions": "Which intent does `utterance` express?",
                "criteria": {f"intent_{i}": description for i, description in enumerate(descriptions)},
            }
        }
    return {
        f"intent_{i}": {"type": "noul", "instructions": f"Does `utterance` express this intent: {description}"}
        for i, description in enumerate(descriptions)
    }


def parse_answers(question_type: QuestionType, answers: Any, n_intents: int) -> list[float]:  # noqa: ANN401
    """Read a response's ``answers`` mapping back into a probability row in description order."""
    if question_type == "choice":
        probabilities = answers[CHOICE_KEY].probabilities
        return [float(probabilities[f"intent_{i}"]) for i in range(n_intents)]
    return [float(answers[f"intent_{i}"].noul) for i in range(n_intents)]


class TypeSafeDescriptionScorer(BaseDescriptionScorer):
    """Zero-shot description scorer backed by TypeSafe's System One model (``jev``).

    Unlike :class:`LLMDescriptionScorer`, which prompts a chat model and buckets its answer,
    this scorer asks the TypeSafe API typed questions and gets calibrated probabilities back:

    - ``question_type="choice"`` (multiclass): one ``Choice`` whose options are the intent
      descriptions; the answer is a distribution over intents.
    - ``question_type="noul"`` (multiclass or multilabel): one ``Noul`` (yes/no) per intent,
      all in one request; each answer is the probability that the utterance expresses it.

    Probabilities are turned into ``log(p)`` (choice) or ``logit(p)`` (noul) "similarities", so
    the base class's softmax / sigmoid with ``temperature=1`` reproduces the model's own
    distribution and ``temperature`` sharpens or flattens it. Answers are cached on disk, so
    repeated predictions on the same utterances (e.g. across HPO trials) cost nothing.

    Requires the ``typesafe`` extra (``pip install "autointent[typesafe]"``) and the
    ``TYPESAFE_API_KEY`` environment variable.

    Args:
        question_type: ``"choice"`` (single question, multiclass only) or ``"noul"`` (one yes/no per intent).
        model: TypeSafe model name; ``None`` uses the SDK default (``jev-latest`` or ``TYPESAFE_DEFAULT_MODEL``).
        temperature: Temperature for scaling the log/logit similarities (default: 1.0).
        max_concurrent: Maximum concurrent requests; ``None`` uses the synchronous client (default: 15).
        max_per_second: Rate limit for requests (default: 10).
        max_retries: Retries per request, delegated to the SDK's ``RetryPolicy`` (default: 3).
        use_cache: Cache answers on disk under the autointent cache dir (default: True).
        multilabel: Flag indicating classification task type.

    Example:
    --------
    .. code-block::

        from autointent.modules.scoring import TypeSafeDescriptionScorer

        scorer = TypeSafeDescriptionScorer(question_type="choice")

        descriptions = [
            "User wants to book or reserve transportation like flights, trains, or hotels",
            "User wants to cancel an existing booking or reservation",
            "User asks about weather conditions or forecasts",
        ]
        scorer.fit([], [], descriptions)

        probabilities = scorer.predict(["Reserve a hotel room", "Delete my booking"])
    """

    name = "description_typesafe"

    def __init__(
        self,
        question_type: QuestionType = "choice",
        model: str | None = None,
        temperature: PositiveFloat = 1.0,
        max_concurrent: PositiveInt | None = 15,
        max_per_second: PositiveInt = 10,
        max_retries: PositiveInt = 3,
        use_cache: bool = True,
        multilabel: bool = False,
    ) -> None:
        super().__init__(temperature=temperature, multilabel=multilabel)
        if question_type == "choice" and multilabel:
            msg = "question_type='choice' picks a single intent and cannot express multilabel targets; use 'noul'"
            raise ValueError(msg)
        self.question_type: QuestionType = question_type
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_per_second = max_per_second
        self.max_retries = max_retries
        self.use_cache = use_cache

    @classmethod
    def from_context(
        cls,
        context: Context,
        question_type: QuestionType | None = None,
        model: str | None = None,
        temperature: PositiveFloat = 1.0,
        max_concurrent: PositiveInt | None = 15,
        max_per_second: PositiveInt = 10,
        max_retries: PositiveInt = 3,
        use_cache: bool = True,
    ) -> TypeSafeDescriptionScorer:
        multilabel = context.is_multilabel()
        if question_type is None:
            question_type = "noul" if multilabel else "choice"
        return cls(
            question_type=question_type,
            model=model,
            temperature=temperature,
            max_concurrent=max_concurrent,
            max_per_second=max_per_second,
            max_retries=max_retries,
            use_cache=use_cache,
            multilabel=multilabel,
        )

    def get_implicit_initialization_params(self) -> dict[str, Any]:
        return {"multilabel": self._multilabel}

    @property
    def resolved_model(self) -> str:
        """Model name actually requested: explicit ``model``, else the SDK's default."""
        default = os.getenv(DEFAULT_MODEL_ENV, DEFAULT_MODEL)
        return self.model or default

    def _fit_implementation(self, descriptions: list[str]) -> None:
        self._description_texts = descriptions
        self._init_runtime()

    def _init_runtime(self) -> None:
        """Create questions, SDK clients, the disk cache and the event loop from the stored config."""
        self._questions = build_questions(self.question_type, self._description_texts)
        self._client, self._async_client = self._create_clients()
        self._cache = PydanticDiskCache(CACHE_SUBDIR, use_cache=self.use_cache)
        self._init_event_loop()

    def _create_clients(self) -> tuple[Any, Any]:
        """Build ``(sync_client, async_client)``.

        The SDK is imported here, not at module level, so the module loads without the
        ``typesafe`` extra; tests patch this method to inject fakes.
        """
        require("typesafe")
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeClient

        retry = RetryPolicy(max_retries=self.max_retries)
        return TypeSafeClient(model=self.model, retry=retry), AsyncTypeSafeClient(model=self.model, retry=retry)

    def _init_event_loop(self) -> None:
        if self.max_concurrent is not None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
            else:
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
            self._event_loop = loop

    def _cache_key(self, utterance: str) -> str:
        hasher = Hasher()
        hasher.update(self.resolved_model)
        hasher.update(self.question_type)
        hasher.update(json.dumps(self._description_texts))
        hasher.update(utterance)
        return hasher.hexdigest()

    def _compute_similarities(self, utterances: list[str]) -> NDArray[np.float64]:
        """Query (or load from cache) one probability row per utterance and map it to similarities."""
        if not (hasattr(self, "_description_texts") and hasattr(self, "_client")):
            msg = "Scorer is not initialized. Call fit() before predict()."
            raise RuntimeError(msg)

        n_intents = len(self._description_texts)
        probabilities = np.full((len(utterances), n_intents), 1.0 / n_intents, dtype=np.float64)

        cache_keys = [self._cache_key(utterance) for utterance in utterances]
        pending: list[int] = []
        for i, cache_key in enumerate(cache_keys):
            cached = self._cache.get_by_key(cache_key, TypeSafeAnswer)
            if cached is None:
                pending.append(i)
            else:
                probabilities[i] = cached.probabilities

        input_tokens = output_tokens = 0
        if pending:
            results = self._ask_many([utterances[i] for i in pending])
            for i, result in zip(pending, results, strict=True):
                if isinstance(result, Exception):
                    logger.warning(
                        "TypeSafe request failed for utterance %r; using uniform scores: %s", utterances[i], result
                    )
                    continue
                row, row_input_tokens, row_output_tokens = result
                probabilities[i] = row
                input_tokens += row_input_tokens
                output_tokens += row_output_tokens
                self._cache.set_by_key(cache_keys[i], TypeSafeAnswer(probabilities=row))

        logger.info(
            "TypeSafe predict: %d utterances, %d from cache, %d requests, %d input tokens, %d output tokens",
            len(utterances),
            len(utterances) - len(pending),
            len(pending),
            input_tokens,
            output_tokens,
        )
        return self._to_similarities(probabilities)

    def _ask_many(self, utterances: list[str]) -> list[_Result]:
        """Send one request per utterance: through aiometer when ``max_concurrent`` is set, else sequentially."""
        if self.max_concurrent is None:
            return [self._ask_one_sync(utterance) for utterance in utterances]
        task = aiometer.run_all(
            [partial(self._ask_one_async, utterance) for utterance in utterances],
            max_at_once=self.max_concurrent,
            max_per_second=self.max_per_second,
        )
        return self._event_loop.run_until_complete(task)

    def _ask_one_sync(self, utterance: str) -> _Result:
        try:
            response = self._client.system_one(state={"utterance": utterance}, questions=self._questions)
        except Exception as e:  # noqa: BLE001  # reason: any SDK/network failure degrades to a uniform row, like the LLM scorer
            return e
        return self._unpack(response)

    async def _ask_one_async(self, utterance: str) -> _Result:
        try:
            response = await self._async_client.system_one(state={"utterance": utterance}, questions=self._questions)
        except Exception as e:  # noqa: BLE001  # reason: any SDK/network failure degrades to a uniform row, like the LLM scorer
            return e
        return self._unpack(response)

    def _unpack(self, response: Any) -> _Result:  # noqa: ANN401
        """Turn an SDK response into ``(row, input_tokens, output_tokens)``; a malformed answer is a failure."""
        try:
            row = parse_answers(self.question_type, response.answers, len(self._description_texts))
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            return e
        usage = getattr(response, "usage", None)
        return row, int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)

    def _to_similarities(self, probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        """``log p`` for choice, ``logit p`` for noul, so the base class's scaling is the identity at T=1."""
        clipped = np.clip(probabilities, _EPS, 1.0 - _EPS)
        if self.question_type == "choice":
            return np.log(clipped)
        return scipy.special.logit(clipped)  # type: ignore[no-any-return]

    def clear_cache(self) -> None:
        """Drop runtime objects (clients, disk-cache handle, event loop)."""
        for attribute in ("_client", "_async_client", "_cache", "_event_loop"):
            if hasattr(self, attribute):
                delattr(self, attribute)
