from __future__ import annotations

import re
from importlib import metadata
from typing import TYPE_CHECKING, get_args

import pytest
from packaging.requirements import Requirement

import autointent._deps as deps

if TYPE_CHECKING:
    from collections.abc import Iterator

_EXTRA_RE = re.compile(r"""extra\s*==\s*['"]([^'"]+)['"]""")


class _FakeMeta:
    def __init__(self, extras: list[str]) -> None:
        self._extras = extras

    def get_all(self, name: str, failobj: list[str] | None = None) -> list[str] | None:
        if name == "Provides-Extra":
            return list(self._extras)
        return failobj


def _patch_metadata(
    monkeypatch: pytest.MonkeyPatch,
    requires_map: dict[str, list[str]],
    versions: dict[str, str],
) -> None:
    """Patch importlib.metadata so deps.* sees a synthetic dependency graph.

    requires_map: {dist_name: [PEP 508 requirement string, ...]}
    versions:     {dist_name: installed_version_string}  (absent key => not installed)
    """
    def fake_requires(dist: str) -> list[str]:
        # Mirror the real importlib.metadata.requires: a dist with no metadata
        # (i.e. not installed) raises PackageNotFoundError rather than returning [].
        # A dist that is installed but has no requirements is modelled by an empty
        # list in requires_map.
        if dist not in requires_map:
            raise metadata.PackageNotFoundError(dist)
        return requires_map[dist]

    def fake_version(name: str) -> str:
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    def fake_metadata(dist: str) -> _FakeMeta:
        extras = sorted({e for s in requires_map.get(dist, []) for e in _EXTRA_RE.findall(s)})
        return _FakeMeta(extras)

    monkeypatch.setattr(metadata, "requires", fake_requires)
    monkeypatch.setattr(metadata, "version", fake_version)
    monkeypatch.setattr(metadata, "metadata", fake_metadata)


def test_check_returns_none_when_satisfied(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metadata(monkeypatch, {}, {"catboost": "1.5.0"})
    assert deps._check(Requirement("catboost>=1.2.8,<2.0.0")) is None


def test_check_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metadata(monkeypatch, {}, {})
    problem = deps._check(Requirement("catboost>=1.2.8"))
    assert problem is not None
    assert "catboost" in problem
    assert "not installed" in problem


def test_check_reports_outdated(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metadata(monkeypatch, {}, {"catboost": "1.0.0"})
    problem = deps._check(Requirement("catboost>=1.2.8,<2.0.0"))
    assert problem is not None
    assert "installed: 1.0.0" in problem


def test_iter_extra_reqs_selects_only_extra_members(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metadata(
        monkeypatch,
        {"autointent": [
            "numpy>=1.0 ; python_version >= '3.0'",          # base dep w/ env marker -> excluded
            "torch>=2.0",                                     # base dep, no marker -> excluded
            "catboost>=1.2.8,<2.0.0 ; extra == 'catboost'",  # extra member -> included
            "peft>=0.10.0 ; extra == 'peft'",                # different extra -> excluded
        ]},
        {},
    )
    reqs = deps._iter_extra_reqs("autointent", "catboost")
    assert {r.name for r in reqs} == {"catboost"}


@pytest.fixture(autouse=True)
def _clear_resolve_cache() -> Iterator[None]:
    deps._resolve_cached.cache_clear()
    yield
    deps._resolve_cached.cache_clear()


def test_resolve_recurses_into_nested_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metadata(
        monkeypatch,
        {
            "autointent": ["transformers[torch]>=4.49.0,<5.0.0 ; extra == 'transformers'"],
            "transformers": [
                "torch>=2.2 ; extra == 'torch'",
                "accelerate>=0.26.0 ; extra == 'torch'",
            ],
        },
        {},
    )
    reqs = deps._resolve("autointent", "transformers", set())
    assert {r.name for r in reqs} == {"transformers", "torch", "accelerate"}


def test_resolve_terminates_on_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metadata(
        monkeypatch,
        {"pkg": [
            "pkg[b]>=1.0 ; extra == 'a'",
            "pkg[a]>=1.0 ; extra == 'b'",
        ]},
        {},
    )
    reqs = deps._resolve("pkg", "a", set())
    assert {r.name for r in reqs} == {"pkg"}


def test_resolve_cached_returns_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metadata(
        monkeypatch,
        {"autointent": ["catboost>=1.2.8 ; extra == 'catboost'"]},
        {},
    )
    result = deps._resolve_cached("autointent", "catboost")
    assert isinstance(result, tuple)
    assert {r.name for r in result} == {"catboost"}


def test_require_passes_when_all_present(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metadata(
        monkeypatch,
        {"autointent": ["catboost>=1.2.8,<2.0.0 ; extra == 'catboost'"]},
        {"catboost": "1.5.0"},
    )
    deps.require("catboost")  # must not raise


def test_require_raises_for_missing_leaf(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metadata(
        monkeypatch,
        {"autointent": ["catboost>=1.2.8,<2.0.0 ; extra == 'catboost'"]},
        {},
    )
    with pytest.raises(ImportError) as exc:
        deps.require("catboost")
    text = str(exc.value)
    assert "catboost" in text
    assert "not installed" in text
    assert "pip install 'autointent[catboost]'" in text


def test_require_raises_for_outdated_version(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metadata(
        monkeypatch,
        {"autointent": ["catboost>=1.2.8,<2.0.0 ; extra == 'catboost'"]},
        {"catboost": "1.0.0"},
    )
    with pytest.raises(ImportError) as exc:
        deps.require("catboost")
    assert "installed: 1.0.0" in str(exc.value)


def test_require_detects_missing_nested_accelerate(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression for #322: accelerate lives in transformers' own [torch] extra,
    # so a transformers-present-but-accelerate-absent env must still be flagged.
    _patch_metadata(
        monkeypatch,
        {
            "autointent": ["transformers[torch]>=4.49.0,<5.0.0 ; extra == 'transformers'"],
            "transformers": [
                "torch>=2.2 ; extra == 'torch'",
                "accelerate>=0.26.0 ; extra == 'torch'",
            ],
        },
        {"transformers": "4.49.0", "torch": "2.2.0"},  # accelerate absent
    )
    with pytest.raises(ImportError) as exc:
        deps.require("transformers")
    assert "accelerate" in str(exc.value)


def test_require_reports_extra_package_entirely_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bare-install path: the extra's top-level package isn't installed at all, so
    # recursing into its nested [torch] extra would read missing metadata. The
    # resolver must NOT leak a raw PackageNotFoundError; instead the parent
    # `transformers[torch]` requirement is flagged as missing with the install hint.
    _patch_metadata(
        monkeypatch,
        {"autointent": ["transformers[torch]>=4.49.0,<5.0.0 ; extra == 'transformers'"]},
        {},  # transformers (and everything else) absent
    )
    with pytest.raises(ImportError) as exc:
        deps.require("transformers")
    text = str(exc.value)
    assert "transformers" in text
    assert "not installed" in text
    assert "pip install 'autointent[transformers]'" in text


def test_iter_extra_reqs_returns_empty_for_uninstalled_dist(monkeypatch: pytest.MonkeyPatch) -> None:
    # The metadata read for a not-installed dist must be swallowed and yield [].
    _patch_metadata(monkeypatch, {}, {})
    assert deps._iter_extra_reqs("not-installed", "torch") == []


def test_require_rejects_unknown_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metadata(
        monkeypatch,
        {"autointent": ["catboost>=1.2.8 ; extra == 'catboost'"]},
        {"catboost": "1.5.0"},
    )
    # "transfomers" is intentionally invalid (typo) to exercise the runtime guard;
    # the type: ignore is required because `Extra` now rejects it at type-check time.
    with pytest.raises(ValueError, match="no extra 'transfomers'"):
        deps.require("transfomers")  # type: ignore[arg-type]


def test_resolve_reads_real_autointent_metadata() -> None:
    # catboost is deliberately chosen: a flat, recursion-free extra, so this real-
    # metadata wiring check is deterministic regardless of what CI installs.
    reqs = deps._resolve_cached("autointent", "catboost")
    assert any(r.name == "catboost" for r in reqs)
    assert all(not r.extras for r in reqs)  # documents the "no nested extra" premise


def test_extra_literal_matches_real_metadata() -> None:
    # The `Extra` Literal is a hand-maintained mirror of the real Provides-Extra
    # metadata. This fails if pyproject gains/loses an extra without the Literal being
    # updated (or vice versa), keeping the static type honest and preventing the
    # manual-sync drift the metadata-driven design otherwise removes.
    assert set(get_args(deps.Extra)) == deps._provides_extras("autointent")


def test_resolve_every_real_extra_without_raising() -> None:
    # Walk every extra autointent actually declares (incl. transformers[torch],
    # which recurses into a nested extra) against real metadata. The resolver must
    # never raise, regardless of which optional packages CI installed -- this is the
    # real-metadata guard for the not-installed-nested-dist fix.
    extras = deps._provides_extras("autointent")
    assert extras  # sanity: metadata wiring returns *something*
    for extra in extras:
        deps._resolve_cached("autointent", extra)  # must not raise
