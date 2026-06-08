"""respx fixture wrapping the OpenAI base URL for httpx-level mocking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import respx

if TYPE_CHECKING:
    from collections.abc import Iterator

    from respx.router import MockRouter


@pytest.fixture
def respx_openai(monkeypatch: pytest.MonkeyPatch) -> Iterator[MockRouter]:
    """Yield a respx.MockRouter scoped to https://api.openai.com.

    Ensures Generator/OpenaiEmbeddingBackend find a valid OPENAI_API_KEY and a model
    name without needing real env vars. Tests register routes on the yielded router.

    OPENAI_BASE_URL is explicitly cleared so a dev-local override (e.g. pointing at a
    local vLLM) doesn't route around respx.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with respx.mock(base_url="https://api.openai.com", assert_all_called=False) as router:
        yield router
