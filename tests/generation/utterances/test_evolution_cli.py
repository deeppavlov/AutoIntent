"""Tests for the ``evolution-aug`` console-script entry point (``_evolution.cli.main``).

Collaborators (dataset loading, the LLM ``Generator``, and both evolver classes)
are mocked at the cli module namespace; the real ``EVOLUTION_MAPPING`` lookup and
argument parsing run for real. No dataset, network, or LLM access.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import autointent.generation.utterances._evolution.cli as evolution_cli

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _patch(monkeypatch: pytest.MonkeyPatch, dataset: MagicMock, augmented: list[str]) -> SimpleNamespace:
    mocks = SimpleNamespace(
        load_dataset=MagicMock(return_value=dataset),
        generator_cls=MagicMock(),
        evolver_cls=MagicMock(),
        incremental_cls=MagicMock(),
    )
    mocks.evolver_cls.return_value.augment.return_value = augmented
    mocks.incremental_cls.return_value.augment.return_value = augmented
    monkeypatch.setattr(evolution_cli, "load_dataset", mocks.load_dataset)
    monkeypatch.setattr(evolution_cli, "Generator", mocks.generator_cls)
    monkeypatch.setattr(evolution_cli, "UtteranceEvolver", mocks.evolver_cls)
    monkeypatch.setattr(evolution_cli, "IncrementalUtteranceEvolver", mocks.incremental_cls)
    return mocks


def test_evolution_cli_regular_evolver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = MagicMock()
    dataset.__getitem__.return_value = ["u1", "u2"]
    mocks = _patch(monkeypatch, dataset, ["new1"])
    out = tmp_path / "out.json"

    monkeypatch.setattr(
        sys,
        "argv",
        ["evolution-aug", "--input-path", "in.json", "--output-path", str(out), "--template", "abstract"],
    )
    evolution_cli.main()

    mocks.evolver_cls.assert_called_once()
    mocks.incremental_cls.assert_not_called()
    mocks.evolver_cls.return_value.augment.assert_called_once()
    dataset.to_json.assert_called_once_with(str(out))
    dataset.push_to_hub.assert_not_called()


def test_evolution_cli_incremental_with_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = MagicMock()
    dataset.__getitem__.return_value = ["u1"]
    mocks = _patch(monkeypatch, dataset, [])
    out = tmp_path / "out.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolution-aug",
            "--input-path",
            "in.json",
            "--output-path",
            str(out),
            "--template",
            "abstract",
            "--decide-for-me",
            "--n-evolutions",
            "2",
            "--sequential",
            "--async-mode",
            "--output-repo",
            "me/repo",
            "--private",
        ],
    )
    evolution_cli.main()

    mocks.incremental_cls.assert_called_once()
    mocks.evolver_cls.assert_not_called()
    augment_kwargs = mocks.incremental_cls.return_value.augment.call_args.kwargs
    assert augment_kwargs["sequential"] is True
    assert augment_kwargs["n_evolutions"] == 2
    dataset.push_to_hub.assert_called_once_with("me/repo", True)
