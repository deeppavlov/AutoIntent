"""Tests for the ``basic-aug`` console-script entry point (``_basic.cli.main``).

The CLI wires together dataset loading, a synthesizer template, and the LLM
``UtteranceGenerator``. Every collaborator is replaced with a mock at the cli
module namespace, so this drives the real argument parsing and control flow
without any dataset, network, or LLM access.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import autointent.generation.utterances._basic.cli as basic_cli

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _patch(monkeypatch: pytest.MonkeyPatch, dataset: MagicMock, augmented: list[str]) -> SimpleNamespace:
    mocks = SimpleNamespace(
        load_dataset=MagicMock(return_value=dataset),
        generator_cls=MagicMock(),
        english_tpl=MagicMock(),
        russian_tpl=MagicMock(),
        utt_gen_cls=MagicMock(),
    )
    mocks.utt_gen_cls.return_value.augment.return_value = augmented
    monkeypatch.setattr(basic_cli, "load_dataset", mocks.load_dataset)
    monkeypatch.setattr(basic_cli, "Generator", mocks.generator_cls)
    monkeypatch.setattr(basic_cli, "EnglishSynthesizerTemplate", mocks.english_tpl)
    monkeypatch.setattr(basic_cli, "RussianSynthesizerTemplate", mocks.russian_tpl)
    monkeypatch.setattr(basic_cli, "UtteranceGenerator", mocks.utt_gen_cls)
    return mocks


def test_basic_cli_english_no_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = MagicMock()
    dataset.__getitem__.return_value = ["u1", "u2", "u3"]
    mocks = _patch(monkeypatch, dataset, ["new1", "new2"])
    out = tmp_path / "out.json"

    monkeypatch.setattr(
        sys,
        "argv",
        ["basic-aug", "--input-path", "in.json", "--output-path", str(out), "--language", "en"],
    )
    basic_cli.main()

    mocks.load_dataset.assert_called_once_with("in.json")
    mocks.english_tpl.assert_called_once()
    mocks.russian_tpl.assert_not_called()
    mocks.utt_gen_cls.return_value.augment.assert_called_once()
    dataset.to_json.assert_called_once_with(str(out))
    dataset.push_to_hub.assert_not_called()


def test_basic_cli_russian_async_with_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = MagicMock()
    dataset.__getitem__.return_value = ["u1"]
    mocks = _patch(monkeypatch, dataset, [])
    out = tmp_path / "out.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basic-aug",
            "--input-path",
            "in.json",
            "--output-path",
            str(out),
            "--language",
            "ru",
            "--output-repo",
            "me/repo",
            "--private",
            "--async-mode",
        ],
    )
    basic_cli.main()

    mocks.russian_tpl.assert_called_once()
    mocks.english_tpl.assert_not_called()
    # the --async-mode flag is forwarded to the generator
    assert mocks.utt_gen_cls.call_args.kwargs["async_mode"] is True
    dataset.push_to_hub.assert_called_once_with("me/repo", private=True)
