"""Mock Generator fixtures for tests where the LLM is incidental.

The real Generator lives at autointent.generation._generator.Generator and is
imported into autointent.modules.scoring._description.llm_encoder. Tests that
want LLMDescriptionScorer to use a mock instead monkeypatch the symbol at the
import site (the scorer rebinds via `from autointent.generation import Generator`
at module load time, so patching the original module is too late).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from autointent.generation import Generator
from autointent.modules.scoring._description.llm_encoder import IntentCategorization


def _make_categorization(most_probable_index: int = 0) -> IntentCategorization:
    """Build a deterministic IntentCategorization with one most-probable intent."""
    return IntentCategorization(
        reasoning="mocked",
        most_probable=[most_probable_index + 1],  # 1-based indices per the schema
        promising=[],
    )


@pytest.fixture
def mock_generator():
    """Return a Mock(spec=Generator) whose sync structured-output returns canned categorization."""
    gen = Mock(spec=Generator)
    gen.get_structured_output_sync.side_effect = lambda _messages, _output_model, _max_retries: _make_categorization(
        _n_intents=10
    )
    gen.get_chat_completion.return_value = "mocked response"
    return gen


@pytest.fixture
def mock_async_generator():
    """Return an AsyncMock-spec'd Generator whose async structured-output returns canned categorization."""
    gen = Mock(spec=Generator)
    gen.get_structured_output_async = AsyncMock(
        side_effect=lambda _messages, _output_model, _max_retries: _make_categorization()
    )
    gen.get_chat_completion_async = AsyncMock(return_value="mocked response")
    return gen


@pytest.fixture
def patch_llm_scorer_generator(monkeypatch):
    """Patch the Generator symbol inside llm_encoder so LLMDescriptionScorer uses the mock.

    Both sync and async code paths on the same instance are exercised; we return a
    factory that returns the same mock_generator instance regardless of constructor args.
    The instance has both sync and async methods configured (see _patched_constructor).
    """
    from autointent.modules.scoring._description import llm_encoder

    combined = Mock(spec=Generator)
    combined.get_structured_output_sync.side_effect = (
        lambda _messages, _output_model, _max_retries: _make_categorization()
    )
    combined.get_structured_output_async = AsyncMock(
        side_effect=lambda _messages, _output_model, _max_retries: _make_categorization()
    )
    combined.get_chat_completion.return_value = "mocked response"
    combined.get_chat_completion_async = AsyncMock(return_value="mocked response")

    def _patched_constructor(*args, **kwargs):
        return combined

    monkeypatch.setattr(llm_encoder, "Generator", _patched_constructor)
    return combined
