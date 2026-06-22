import re
from importlib import metadata

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
