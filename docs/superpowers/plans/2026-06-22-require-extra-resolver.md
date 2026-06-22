# Metadata-driven `require(extra)` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the import-based `require(dependency, extra)` guard with `require(extra)`, which reads installed package metadata to verify every dependency of an autointent extra — recursively, including nested extras like `transformers[torch] → accelerate` — is installed and version-satisfied.

**Architecture:** A new focused module `src/autointent/_deps.py` resolves an extra's requirement graph via `importlib.metadata` (the build bakes `pyproject.toml`'s requirements into the wheel as metadata; the source file is not shipped) and validates each leaf with `packaging`. `require` moves there; all ~20 call sites pass just the extra name. Fixes #322 structurally — the missing `accelerate` is caught by recursing into `transformers`' own `torch` extra, with no hand-added guard.

**Tech Stack:** Python 3.10+, `importlib.metadata` (stdlib), `packaging` (promoted to a core dependency), pytest with `monkeypatch`.

## Global Constraints

- **Lint:** `ruff` runs with `select = ["ALL"]`, line-length 120, target `py310`, **google docstring convention**. Every module/function needs a google-style docstring; every parameter and return needs a type annotation. Raise exceptions via a pre-assigned `msg` variable (satisfies `EM`/`TRY003`).
- **Types:** `mypy --strict` on **python 3.10**. `from __future__ import annotations` at the top of every new `.py`. When building a typed `set[str]`/`dict` from `packaging.utils.canonicalize_name` (which returns `NormalizedName`), wrap in `str(...)` — `set` is invariant so `set[NormalizedName]` is not assignable to `set[str]`.
- **New core dependency:** add `packaging (>=23.2)` to `[project].dependencies` (no upper cap — avoids resolver conflicts; it is currently present only transitively, the same mistake #322 is about).
- **Testing:** Do **NOT** run the full pytest suite locally — it can freeze the machine. The new `tests/test_deps.py` is pure Python (monkeypatched metadata, no model loading) and is the **only** suite you may run locally, via `pytest tests/test_deps.py -q`. Full-suite + call-site migration verification goes through **CI** (push branch + coverage dispatch). `ruff` and `mypy` may be run locally.
- **Do not** commit `uv.lock` (gitignored by design; CI re-solves deps).
- **Extra names are passed verbatim as declared in `pyproject.toml`** (e.g. `"sentence-transformers"` with a hyphen).

---

### Task 1: Create `_deps.py` with `_check` + add `packaging` core dep

**Files:**
- Modify: `pyproject.toml` (add `packaging` to `[project].dependencies`)
- Create: `src/autointent/_deps.py`
- Create: `tests/test_deps.py`

**Interfaces:**
- Produces: `_check(req: packaging.requirements.Requirement) -> str | None` — returns a one-line problem description if the dist named by `req` is not installed or its installed version is outside `req.specifier`, else `None`.

- [ ] **Step 1: Add `packaging` to core dependencies**

In `pyproject.toml`, inside `[project]` `dependencies = [ ... ]`, add one line (keep existing entries):

```toml
    "packaging (>=23.2)",
```

- [ ] **Step 2: Create the module with its docstring, imports, and `_check`**

Create `src/autointent/_deps.py`:

```python
"""Validate optional-extra dependencies from installed package metadata.

The :func:`require` guard checks that every dependency of an ``autointent`` extra
is installed and version-satisfied. It reads the metadata that the build baked into
the installed distribution (via :mod:`importlib.metadata`) rather than the source
``pyproject.toml``, which is not shipped in the wheel. Nested extras are resolved
recursively, so e.g. the ``transformers`` extra (``transformers[torch]``)
transitively requires ``accelerate`` and that is checked too.
"""

from __future__ import annotations

from importlib import metadata

from packaging.requirements import Requirement

_DIST = "autointent"


def _check(req: Requirement) -> str | None:
    """Check a single requirement against the installed environment.

    Args:
        req: The parsed requirement to validate.

    Returns:
        A human-readable problem description if the distribution is missing or its
        installed version does not satisfy ``req.specifier``; ``None`` otherwise.
    """
    try:
        installed = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        return f"{req.name}{req.specifier} (not installed)"
    if req.specifier and not req.specifier.contains(installed, prereleases=True):
        return f"{req.name}{req.specifier} (installed: {installed})"
    return None
```

- [ ] **Step 3: Write the failing test and shared metadata-faking helper**

Create `tests/test_deps.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_deps.py -q`
Expected: 3 passed.

- [ ] **Step 5: Lint and type-check**

Run: `ruff check src/autointent/_deps.py tests/test_deps.py && mypy src/autointent/_deps.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/autointent/_deps.py tests/test_deps.py
git commit -m "feat(deps): add _check leaf version validator + packaging core dep (#322)"
```

---

### Task 2: `_iter_extra_reqs` — read a dist's requirements for one extra

**Files:**
- Modify: `src/autointent/_deps.py`
- Test: `tests/test_deps.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_iter_extra_reqs(dist: str, extra: str) -> list[Requirement]` — the requirements of `dist` activated by `extra` for the current environment. Base dependencies (no extra marker, or only an environment marker) are excluded.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_deps.py`:

```python
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
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pytest tests/test_deps.py::test_iter_extra_reqs_selects_only_extra_members -q`
Expected: FAIL with `AttributeError: module 'autointent._deps' has no attribute '_iter_extra_reqs'`.

- [ ] **Step 3: Implement `_iter_extra_reqs`**

In `src/autointent/_deps.py`, add `from packaging.utils import canonicalize_name` to the imports (keep the import block sorted: `Requirement` then `canonicalize_name`), and append:

```python
def _iter_extra_reqs(dist: str, extra: str) -> list[Requirement]:
    """Return the requirements of ``dist`` activated by ``extra``.

    A requirement is included only when its marker is satisfied *because* of the
    extra: it must evaluate true with the extra set and false with no extra. This
    excludes base dependencies that merely carry an environment marker.

    Args:
        dist: Distribution name whose metadata is read.
        extra: Extra name whose dependencies are wanted.

    Returns:
        The parsed requirements activated by ``extra`` in the current environment.
    """
    target = str(canonicalize_name(extra))
    result: list[Requirement] = []
    for spec in metadata.requires(dist) or []:
        req = Requirement(spec)
        marker = req.marker
        if marker is None:
            continue
        if marker.evaluate({"extra": target}) and not marker.evaluate({"extra": ""}):
            result.append(req)
    return result
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_deps.py -q`
Expected: all passed.

- [ ] **Step 5: Lint and type-check**

Run: `ruff check src/autointent/_deps.py tests/test_deps.py && mypy src/autointent/_deps.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/autointent/_deps.py tests/test_deps.py
git commit -m "feat(deps): resolve a dist's requirements for a single extra"
```

---

### Task 3: `_resolve` — recursive walk with cycle guard + cached entry point

**Files:**
- Modify: `src/autointent/_deps.py`
- Test: `tests/test_deps.py`

**Interfaces:**
- Consumes: `_iter_extra_reqs`.
- Produces:
  - `_resolve(dist: str, extra: str, seen: set[tuple[str, str]]) -> list[Requirement]` — every leaf requirement reachable from `dist[extra]`, recursing through nested extras, cycle-guarded by `seen`.
  - `_resolve_cached(dist: str, extra: str) -> tuple[Requirement, ...]` — memoized wrapper (call this from `require`).

- [ ] **Step 1: Write the failing tests + cache-clear fixture**

First add `import pytest` to the **third-party** import group of `tests/test_deps.py` — it must come *before* `from packaging.requirements import Requirement` (ruff's isort sorts straight `import` statements before `from` imports within a section). Then add the fixture and tests (the autouse fixture keeps the cache from leaking synthetic graphs across tests):

```python
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
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `pytest tests/test_deps.py -k resolve -q`
Expected: FAIL with `AttributeError: module 'autointent._deps' has no attribute '_resolve'`.

- [ ] **Step 3: Implement `_resolve` and `_resolve_cached`**

In `src/autointent/_deps.py`, add `from functools import cache` to the imports (above `from importlib import metadata`), and append (use `@cache`, not `@lru_cache(maxsize=None)` — ruff `ALL` raises `UP033` for the latter; `functools.cache` exists on py3.10 and still exposes `.cache_clear()`):

```python
def _resolve(dist: str, extra: str, seen: set[tuple[str, str]]) -> list[Requirement]:
    """Recursively collect every leaf requirement activated by ``dist[extra]``.

    Each activated requirement is returned for version checking, and any nested
    extras it declares (e.g. ``transformers[torch]``) are resolved in turn.

    Args:
        dist: Distribution name to start from.
        extra: Extra name to resolve.
        seen: Visited ``(dist, extra)`` pairs, used to break dependency cycles.

    Returns:
        The flattened list of requirements to validate.
    """
    key = (str(canonicalize_name(dist)), str(canonicalize_name(extra)))
    if key in seen:
        return []
    seen.add(key)

    leaves: list[Requirement] = []
    for req in _iter_extra_reqs(dist, extra):
        leaves.append(req)
        for nested in req.extras:
            leaves.extend(_resolve(req.name, nested, seen))
    return leaves


@cache
def _resolve_cached(dist: str, extra: str) -> tuple[Requirement, ...]:
    """Memoized :func:`_resolve`; the metadata graph shape is stable per process.

    Args:
        dist: Distribution name to start from.
        extra: Extra name to resolve.

    Returns:
        The resolved requirements as an immutable tuple.
    """
    return tuple(_resolve(dist, extra, set()))
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_deps.py -q`
Expected: all passed.

- [ ] **Step 5: Lint and type-check**

Run: `ruff check src/autointent/_deps.py tests/test_deps.py && mypy src/autointent/_deps.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/autointent/_deps.py tests/test_deps.py
git commit -m "feat(deps): recursive extra resolution with cycle guard + cache"
```

---

### Task 4: `require` — public guard (unknown-extra check, aggregation, message)

**Files:**
- Modify: `src/autointent/_deps.py`
- Test: `tests/test_deps.py`

**Interfaces:**
- Consumes: `_resolve_cached`, `_check`.
- Produces: `require(extra: str, *, dist: str = "autointent") -> None`. Raises `ValueError` if `dist` declares no such extra (typo guard); raises `ImportError` listing every missing/outdated dependency with a `pip install '<dist>[<extra>]'` hint; otherwise returns `None`. This is the symbol every call site imports.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_deps.py`:

```python
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
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `pytest tests/test_deps.py -k require -q`
Expected: FAIL with `AttributeError: module 'autointent._deps' has no attribute 'require'`.

- [ ] **Step 3: Implement `_provides_extras` and `require`**

In `src/autointent/_deps.py`, append:

```python
def _provides_extras(dist: str) -> set[str]:
    """Return the normalized set of extras declared by ``dist``.

    Args:
        dist: Distribution name whose metadata is read.

    Returns:
        Normalized extra names from the distribution's ``Provides-Extra`` metadata.
    """
    md = metadata.metadata(dist)
    return {str(canonicalize_name(e)) for e in (md.get_all("Provides-Extra") or [])}


def require(extra: str, *, dist: str = _DIST) -> None:
    """Ensure every dependency of an ``autointent`` extra is installed and current.

    Args:
        extra: The extra to validate, e.g. ``"transformers"``.
        dist: Distribution that declares the extra. Defaults to ``"autointent"``.

    Raises:
        ValueError: If ``dist`` declares no such ``extra`` (typically a typo).
        ImportError: If any required dependency is missing or its installed version
            does not satisfy the constraint declared in the metadata.
    """
    known = _provides_extras(dist)
    if str(canonicalize_name(extra)) not in known:
        msg = f"'{dist}' declares no extra '{extra}'. Known extras: {', '.join(sorted(known))}."
        raise ValueError(msg)

    problems: list[str] = []
    for req in _resolve_cached(dist, extra):
        problem = _check(req)
        if problem is not None and problem not in problems:
            problems.append(problem)

    if problems:
        bullets = "\n".join(f"  - {p}" for p in problems)
        msg = (
            f"Feature requires extra '{extra}', but dependencies are missing or outdated:\n"
            f"{bullets}\n"
            f"Install with: pip install '{dist}[{extra}]'"
        )
        raise ImportError(msg)
```

- [ ] **Step 4: Run the full module test file**

Run: `pytest tests/test_deps.py -q`
Expected: all passed.

- [ ] **Step 5: Lint and type-check**

Run: `ruff check src/autointent/_deps.py tests/test_deps.py && mypy src/autointent/_deps.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/autointent/_deps.py tests/test_deps.py
git commit -m "feat(deps): add require(extra) guard with aggregated, versioned errors (#322)"
```

---

### Task 5: Migrate all call sites + remove old `require` from `_utils.py`

**Files:**
- Modify: `src/autointent/_utils.py` (remove old `require` + now-unused `import importlib`)
- Modify: `user_guides/advanced/02_embedder_configuration.py` (stale prose reference to `autointent._utils.require`)
- Modify (import line `from autointent._utils import require` → `from autointent._deps import require`, and each `require(...)` call):
  - `src/autointent/modules/scoring/_bert.py`
  - `src/autointent/modules/scoring/_ptuning/ptuning.py`
  - `src/autointent/modules/scoring/_lora/lora.py`
  - `src/autointent/modules/scoring/_catboost/catboost_scorer.py`
  - `src/autointent/_wrappers/embedder/vllm.py`
  - `src/autointent/_wrappers/embedder/openai.py`
  - `src/autointent/_wrappers/embedder/sentence_transformers.py`
  - `src/autointent/_wrappers/ranker.py`
  - `src/autointent/generation/_generator.py`
  - `src/autointent/_dump_tools/unit_dumpers.py`

**Interfaces:**
- Consumes: `require(extra: str)` from `autointent._deps` (Task 4).
- Produces: no new symbols. After this task no module imports `require` from `autointent._utils`, and no `require(` call passes more than one argument.

- [ ] **Step 1: Remove the old `require` and its import from `_utils.py`**

In `src/autointent/_utils.py`: delete the entire `require` function (the `def require(...)` block and its body) and delete the top-level `import importlib` line (it was only used by `require`). Leave `import torch`, `from typing import TypeVar`, `_funcs_to_dict`, and `detect_device` untouched.

- [ ] **Step 2: Update every import line**

In each of the 10 files listed above, change:

```python
from autointent._utils import require
```

to:

```python
from autointent._deps import require
```

- [ ] **Step 3: Update the call sites (one argument = the extra name)**

Apply these exact replacements:

- `src/autointent/modules/scoring/_bert.py`: `require("transformers", "transformers")` → `require("transformers")`
- `src/autointent/modules/scoring/_ptuning/ptuning.py`: `require("peft", extra="peft")` → `require("peft")`
- `src/autointent/modules/scoring/_lora/lora.py`: `require("peft", extra="peft")` → `require("peft")`
- `src/autointent/modules/scoring/_catboost/catboost_scorer.py`: `require("catboost", extra="catboost")` → `require("catboost")`
- `src/autointent/_wrappers/embedder/vllm.py`: `require("vllm", extra="vllm")` → `require("vllm")`
- `src/autointent/_wrappers/embedder/openai.py`: `require("tiktoken", "openai")` → `require("openai")` **and** `require("openai", "openai")` → `require("openai")`
- `src/autointent/generation/_generator.py`: `require("openai", "openai")` → `require("openai")`
- `src/autointent/_wrappers/ranker.py`: `require("sentence_transformers", extra="sentence-transformers")` → `require("sentence-transformers")`
- `src/autointent/_wrappers/embedder/sentence_transformers.py`:
  - line ~45: `require("transformers", extra="transformers")` → `require("transformers")`
  - line ~133: `require("sentence_transformers", extra="sentence-transformers")` → `require("sentence-transformers")`
  - the training block (three consecutive calls):
    ```python
    require("sentence_transformers", extra="sentence-transformers")
    require("transformers", extra="transformers")
    require("accelerate", extra="transformers")
    ```
    becomes (drop the `accelerate` line — it is now covered by recursing into the `transformers` extra):
    ```python
    require("sentence-transformers")
    require("transformers")
    ```
- `src/autointent/_dump_tools/unit_dumpers.py` — replace all occurrences:
  - `require("peft", extra="peft")` → `require("peft")`
  - `require("transformers", extra="transformers")` → `require("transformers")`
  - `require("catboost", extra="catboost")` → `require("catboost")`

Then fix the stale prose reference in `user_guides/advanced/02_embedder_configuration.py` (line ~22): change `autointent._utils.require` to `autointent._deps.require` (it is documentation prose, not an import — no lint impact, just keeping the docs accurate).

- [ ] **Step 4: Verify no stale call shape or import remains**

Run:
```bash
grep -rn 'from autointent._utils import require' src/autointent --include='*.py'; \
grep -rnE 'require\([^)]*,' src/autointent --include='*.py' | grep -v 'def require'
```
Expected: **no output** from either grep (no old import; no `require(` call with a second argument).

- [ ] **Step 5: Lint and type-check the whole package**

Run: `ruff check src/autointent && mypy src/autointent`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/autointent
git commit -m "refactor(deps): migrate all require() call sites to extra-name guard (#322)"
```

---

### Task 6: Real-metadata smoke test + push to CI

**Files:**
- Modify: `tests/test_deps.py` (add one un-monkeypatched wiring test)

**Interfaces:**
- Consumes: `_resolve_cached` against the real installed `autointent` metadata.
- Produces: no new symbols.

- [ ] **Step 1: Add a real-metadata wiring test**

Add to `tests/test_deps.py` (no `monkeypatch` — exercises the real `importlib.metadata` reads to catch wiring regressions; asserts only resolution shape, not install state, so it is deterministic regardless of which extras CI installs):

```python
def test_resolve_reads_real_autointent_metadata():
    reqs = deps._resolve_cached("autointent", "catboost")
    assert any(r.name == "catboost" for r in reqs)
```

- [ ] **Step 2: Run the isolated test file**

Run: `pytest tests/test_deps.py -q`
Expected: all passed.

- [ ] **Step 3: Lint and type-check**

Run: `ruff check src/autointent tests/test_deps.py && mypy src/autointent`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_deps.py
git commit -m "test(deps): smoke-test require resolution against real metadata"
```

- [ ] **Step 5: Push the branch and trigger CI**

Push the working branch and open a PR (or trigger the coverage dispatch) so the **full** suite — including the migrated call-site modules — runs in CI. Do not run the full suite locally. Confirm CI is green before considering the work done; link the run in the PR referencing #322.

---

## Notes / deviations from the spec

- **No re-export from `_utils`.** The spec proposed re-exporting `require` from `autointent._utils` to keep import lines untouched. Under `ruff select = ["ALL"]` a bare re-export trips `F401`, and the redundant-alias workaround trips `PLC0414`. Since every call-site file is edited for the signature change anyway, updating its import line to `from autointent._deps import require` is cleaner and lint-clean. The dependency logic now has a single home (`_deps.py`).
- **Cycle-guard key type.** The spec sketched `seen: set[tuple[str, frozenset[str]]]`. The plan uses `set[tuple[str, str]]` because `_resolve` recurses one nested-extra name at a time (it iterates `req.extras` and recurses per name), so the key is `(dist, single-extra-name)`. Behaviourally equivalent for cycle-breaking; simpler.
- **Error-bullet ordering.** Problems are listed in resolution order (e.g. `transformers` before its nested `accelerate`), which is the reverse of the illustrative example in the spec. Cosmetic — no test asserts bullet order.

## Self-review

- **Spec coverage:** new `_deps.py` (Tasks 1–4); metadata-not-pyproject premise (module docstring, Task 1); metadata version check (`_check`, Task 1); recursive nested-extra resolution incl. accelerate/#322 (Task 3 + Task 4 regression test); aggregated actionable error (Task 4); `packaging` promoted to core dep (Task 1); all ~20 call sites migrated + clean replace (Task 5); synthetic-metadata TDD for every scenario incl. cycle/unknown-extra (Tasks 1–4); real-metadata smoke test (Task 6); CI verification per project convention (Task 6, Global Constraints). The one spec deviation (re-export) is documented above.
- **Placeholder scan:** none — every code/test step contains complete code; every command has expected output.
- **Type consistency:** `_check`, `_iter_extra_reqs`, `_resolve`, `_resolve_cached`, `_provides_extras`, `require` keep identical signatures across the tasks that define and call them; `_resolve_cached` (not `_resolve`) is the symbol `require` consumes, matching Task 3's "Produces" and Task 4's "Consumes".
