import re
from importlib import metadata

import pytest
from packaging.requirements import Requirement

import autointent._deps as deps

_EXTRA_RE = re.compile(r"""extra\s*==\s*['"]([^'"]+)['"]""")


class _FakeMeta:
    def __init__(self, extras):
        self._extras = extras

    def get_all(self, name, failobj=None):
        if name == "Provides-Extra":
            return list(self._extras)
        return failobj


def _patch_metadata(monkeypatch, requires_map, versions):
    """Patch importlib.metadata so deps.* sees a synthetic dependency graph.

    requires_map: {dist_name: [PEP 508 requirement string, ...]}
    versions:     {dist_name: installed_version_string}  (absent key => not installed)
    """
    def fake_requires(dist):
        return requires_map.get(dist, [])

    def fake_version(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    def fake_metadata(dist):
        extras = sorted({e for s in requires_map.get(dist, []) for e in _EXTRA_RE.findall(s)})
        return _FakeMeta(extras)

    monkeypatch.setattr(deps.metadata, "requires", fake_requires)
    monkeypatch.setattr(deps.metadata, "version", fake_version)
    monkeypatch.setattr(deps.metadata, "metadata", fake_metadata)


def test_check_returns_none_when_satisfied(monkeypatch):
    _patch_metadata(monkeypatch, {}, {"catboost": "1.5.0"})
    assert deps._check(Requirement("catboost>=1.2.8,<2.0.0")) is None


def test_check_reports_missing(monkeypatch):
    _patch_metadata(monkeypatch, {}, {})
    problem = deps._check(Requirement("catboost>=1.2.8"))
    assert problem is not None
    assert "catboost" in problem
    assert "not installed" in problem


def test_check_reports_outdated(monkeypatch):
    _patch_metadata(monkeypatch, {}, {"catboost": "1.0.0"})
    problem = deps._check(Requirement("catboost>=1.2.8,<2.0.0"))
    assert problem is not None
    assert "installed: 1.0.0" in problem


def test_iter_extra_reqs_selects_only_extra_members(monkeypatch):
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
def _clear_resolve_cache():
    deps._resolve_cached.cache_clear()
    yield
    deps._resolve_cached.cache_clear()


def test_resolve_recurses_into_nested_extra(monkeypatch):
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


def test_resolve_terminates_on_cycle(monkeypatch):
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


def test_resolve_cached_returns_tuple(monkeypatch):
    _patch_metadata(
        monkeypatch,
        {"autointent": ["catboost>=1.2.8 ; extra == 'catboost'"]},
        {},
    )
    result = deps._resolve_cached("autointent", "catboost")
    assert isinstance(result, tuple)
    assert {r.name for r in result} == {"catboost"}


def test_require_passes_when_all_present(monkeypatch):
    _patch_metadata(
        monkeypatch,
        {"autointent": ["catboost>=1.2.8,<2.0.0 ; extra == 'catboost'"]},
        {"catboost": "1.5.0"},
    )
    deps.require("catboost")  # must not raise


def test_require_raises_for_missing_leaf(monkeypatch):
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


def test_require_raises_for_outdated_version(monkeypatch):
    _patch_metadata(
        monkeypatch,
        {"autointent": ["catboost>=1.2.8,<2.0.0 ; extra == 'catboost'"]},
        {"catboost": "1.0.0"},
    )
    with pytest.raises(ImportError) as exc:
        deps.require("catboost")
    assert "installed: 1.0.0" in str(exc.value)


def test_require_detects_missing_nested_accelerate(monkeypatch):
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


def test_require_rejects_unknown_extra(monkeypatch):
    _patch_metadata(
        monkeypatch,
        {"autointent": ["catboost>=1.2.8 ; extra == 'catboost'"]},
        {"catboost": "1.5.0"},
    )
    with pytest.raises(ValueError, match="no extra 'transfomers'"):
        deps.require("transfomers")  # typo
