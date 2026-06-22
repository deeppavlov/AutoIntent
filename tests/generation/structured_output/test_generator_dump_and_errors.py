"""Tests for Generator serialization and chat-completion error handling.

These cover paths the existing structured-output tests don't: the dump/load
round-trip and the async empty-response guard. The OpenAI client is constructed
for real (no network) via the respx fixture, which supplies a fake API key and
model name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

pytest.importorskip("openai")

from autointent.generation import Generator
from autointent.generation.chat_templates import Role

if TYPE_CHECKING:
    from pathlib import Path

    from respx.router import MockRouter


def _empty_choices_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
        },
    )


def test_generator_dump_load_round_trip(respx_openai: MockRouter, tmp_path: Path) -> None:
    gen = Generator(model_name="gpt-test", base_url=None, use_cache=False, temperature=0.5, max_tokens=128)

    gen.dump(tmp_path / "gen")
    loaded = Generator.load(tmp_path / "gen")

    assert loaded.model_name == "gpt-test"
    assert loaded.base_url is None
    assert loaded.use_cache is False
    assert loaded.generation_params == {"temperature": 0.5, "max_tokens": 128}


@pytest.mark.asyncio
async def test_get_chat_completion_async_raises_on_empty_choices(respx_openai: MockRouter) -> None:
    generator = Generator(use_cache=False)
    respx_openai.post("/v1/chat/completions").mock(return_value=_empty_choices_response())

    with pytest.raises(RuntimeError, match="No response received"):
        await generator.get_chat_completion_async([{"role": Role.USER, "content": "hi"}])
