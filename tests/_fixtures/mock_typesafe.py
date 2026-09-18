"""Fake TypeSafe clients for tests where the API is incidental.

``TypeSafeDescriptionScorer`` builds its SDK clients lazily in ``_create_clients`` so the
module imports without the ``typesafe`` extra. The fixture below patches that method to
return fakes whose ``system_one`` answers deterministically from the question dicts, and
redirects the autointent disk cache to ``tmp_path`` so runs never see each other's entries.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from autointent.modules.scoring._description.typesafe import CHOICE_KEY, TypeSafeDescriptionScorer

if TYPE_CHECKING:
    from pathlib import Path

BEST_PROBABILITY = 0.7
NOUL_YES = 0.9
NOUL_NO = 0.1


def fake_response(questions: dict[str, dict[str, Any]], utterance: str, best_index: int = 0) -> SimpleNamespace:
    """Deterministic answers: option ``best_index`` gets 0.7 and the rest share 0.3 (choice), or 0.9 vs 0.1 (noul).

    ``utterance`` is accepted so callers can build the expectation for a specific input; the
    fake ignores its content.
    """
    del utterance
    answers: dict[str, SimpleNamespace] = {}
    if CHOICE_KEY in questions and questions[CHOICE_KEY]["type"] == "choice":
        keys = list(questions[CHOICE_KEY]["criteria"])
        rest = (1.0 - BEST_PROBABILITY) / max(len(keys) - 1, 1)
        probabilities = {key: (BEST_PROBABILITY if i == best_index else rest) for i, key in enumerate(keys)}
        answers[CHOICE_KEY] = SimpleNamespace(choice=keys[best_index], confidence=0.5, probabilities=probabilities)
    else:
        for i, key in enumerate(questions):
            answers[key] = SimpleNamespace(noul=NOUL_YES if i == best_index else NOUL_NO)
    return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=100, output_tokens=10), model="jev-test")


class FakeTypeSafeClient:
    """Sync stand-in for ``typesafe_sdk.TypeSafeClient`` recording every utterance it was asked about."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def system_one(self, state: dict[str, Any], questions: dict[str, dict[str, Any]], **_: Any) -> SimpleNamespace:
        self.calls.append(state["utterance"])
        return fake_response(questions, state["utterance"])


class FakeAsyncTypeSafeClient(FakeTypeSafeClient):
    """Async stand-in for ``typesafe_sdk.AsyncTypeSafeClient``."""

    async def system_one(  # mypy's override check is disabled project-wide (disable_error_code)
        self, state: dict[str, Any], questions: dict[str, dict[str, Any]], **_: Any
    ) -> SimpleNamespace:
        return FakeTypeSafeClient.system_one(self, state, questions)


@pytest.fixture
def patch_typesafe_scorer_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[FakeTypeSafeClient, FakeAsyncTypeSafeClient]:
    """Make every ``TypeSafeDescriptionScorer`` use the fakes and an isolated disk cache."""
    monkeypatch.setattr("autointent.generation._cache.user_cache_dir", lambda *_: str(tmp_path / "cache"))
    sync_client, async_client = FakeTypeSafeClient(), FakeAsyncTypeSafeClient()
    monkeypatch.setattr(TypeSafeDescriptionScorer, "_create_clients", lambda _self: (sync_client, async_client))
    return sync_client, async_client
