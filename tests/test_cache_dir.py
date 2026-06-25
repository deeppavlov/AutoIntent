from __future__ import annotations

from typing import TYPE_CHECKING

from autointent._cache_dir import get_cache_dir

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_get_cache_dir_honors_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOINTENT_CACHE_DIR", str(tmp_path / "custom"))
    assert get_cache_dir() == tmp_path / "custom"


def test_get_cache_dir_falls_back_to_appdirs(monkeypatch: pytest.MonkeyPatch) -> None:
    # The global autouse isolation fixture sets the env var for every test, so unset it here.
    monkeypatch.delenv("AUTOINTENT_CACHE_DIR", raising=False)
    result = get_cache_dir()
    assert result.name == "autointent" or "autointent" in str(result)
