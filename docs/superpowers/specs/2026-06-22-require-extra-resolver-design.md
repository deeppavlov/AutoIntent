# Design: metadata-driven `require(extra)` dependency guard

**Date:** 2026-06-22
**Issue:** [#322](https://github.com/voorhs/AutoIntent/issues/322) — bert scorer: opaque ImportError when `accelerate` is missing (require() guard checks only `transformers`)
**Status:** Approved, pending implementation plan

## Problem

The current `require(dependency, extra=None)` utility (`src/autointent/_utils.py`) tries to
`importlib.import_module(dependency)` and raises an informative `ImportError` naming the pip
extra if the import fails. Two structural weaknesses:

1. **Manual sync burden.** Each `require` call site hard-codes a module name and an extra label.
   These must be kept in sync with `pyproject.toml` by hand. When an extra gains a dependency,
   every relevant call site must be updated, and it is easy to miss one.
2. **No version checking.** It only checks *presence* (importability), never whether the installed
   version satisfies the constraint declared in `pyproject.toml`.

Issue #322 is the concrete failure: `BertScorer.__init__` calls `require("transformers", "transformers")`,
which passes whenever `transformers` is importable — even when `accelerate` (pulled by the
`transformers[torch]` extra, required by HF `Trainer`) is absent. The user then hits a raw, deep
`ImportError` from inside `Trainer` instead of AutoIntent's "install `autointent[transformers]`" message.

## Goal

Refactor `require` to take an **autointent extra name** and validate that **every** dependency of
that extra — recursively, including nested third-party extras such as `transformers[torch] → accelerate`
— is installed **and** version-satisfied, raising a single aggregated, actionable `ImportError`
otherwise.

This removes the manual sync burden (single source of truth = the package metadata baked from
`pyproject.toml`), adds version checking, and fixes #322 structurally — with no hand-added
`accelerate` guard.

## Key technical premise

`pyproject.toml` is **not** shipped in the wheel and cannot be parsed at runtime. It does not need to
be: the build process bakes the same requirements (with version specifiers and `extra` markers) into
the installed package metadata, which `importlib.metadata` reads at runtime. Verified against the
installed `autointent`:

```
transformers[torch]>=4.49.0,<5.0.0 ; extra == 'transformers'
```

and, recursing into `transformers`' own metadata:

```
torch>=2.2 ; extra == "torch"
accelerate>=0.26.0 ; extra == "torch"
```

So the single source of truth is the installed metadata, available everywhere the package is installed.

## Decisions (from brainstorming Q&A)

- **Presence check method:** metadata version check via `importlib.metadata` (not import-based). Faster,
  no heavy imports (e.g. torch), enables version checking and recursive extra resolution.
- **Resolution depth:** full recursive walk of nested extras, cycle-guarded. This is what catches the
  nested `accelerate` requirement automatically.
- **Rollout:** clean replace — change the signature to `require(extra: str)` and migrate every call
  site in this PR. `require` is internal (`autointent._utils`, ~20 call sites, no external contract).

## Design

### New module `src/autointent/_deps.py`

Houses the dependency-resolution logic, keeping it out of `_utils.py` (which imports `torch` at module
load) and making it trivially unit-testable.

- **`require(extra: str, *, dist: str = "autointent") -> None`** — public entry point. Resolves the
  extra's full (recursive) leaf-requirement set and validates each. Raises `ImportError` with an
  aggregated message if anything is missing or outdated; returns `None` otherwise.

- **`_iter_extra_reqs(dist, extra)`** — read `importlib.metadata.requires(dist)` (treat `None` as
  empty), parse each entry with `packaging.requirements.Requirement`, and keep those whose `.marker`
  evaluates true for `{"extra": extra}` combined with the current environment. `packaging` fills
  environment markers (`python_version`, `sys_platform`, …) from the running interpreter, so
  platform-conditional dependencies resolve correctly.

- **`_resolve(dist, extra, seen)`** — yields leaf requirements (those with no further extras) reachable
  from `dist`'s `extra`, recursing into nested extras. For a requirement like
  `transformers[torch]>=4.49.0,<5.0.0` it yields the `transformers` constraint itself and then recurses
  into `transformers`' `torch` extra. Cycle-guarded via `seen: set[tuple[str, frozenset[str]]]`.

- **`_check(req) -> str | None`** — `importlib.metadata.version(req.name)`; return a human-readable
  problem string if the dist is absent (`PackageNotFoundError`) or the installed version is not in
  `req.specifier`; otherwise `None`.

- Distribution/extra names normalized with `packaging.utils.canonicalize_name` on both sides for
  hyphen/underscore robustness.

- `_utils.py` re-exports `require` (`from autointent._deps import require`) so existing
  `from autointent._utils import require` import lines stay untouched.

- **Caching:** memoize the pure resolution step (extra → tuple of leaf requirements) with
  `functools.lru_cache` keyed by `(dist, extra)`, so hot module-init paths do not re-walk metadata.
  Version checks run on each call (cheap).

### The #322 mechanism, concretely

`transformers[torch]>=4.49.0,<5.0.0` parses to `name="transformers"`, `extras={"torch"}`, plus the
specifier. `_resolve` checks the `transformers` version against the specifier, then recurses via
`_iter_extra_reqs("transformers", "torch")` → `accelerate>=0.26.0`, `torch>=2.2`. A missing `accelerate`
surfaces here, named explicitly, with the `autointent[transformers]` install hint.

### Error shape (aggregated)

```
ImportError: Feature requires extra 'transformers', but dependencies are missing or outdated:
  - accelerate>=0.26.0 (not installed)
  - transformers>=4.49.0,<5.0.0 (installed: 4.30.0)
Install with: pip install 'autointent[transformers]'
```

### pyproject change

Add `packaging` to **core** `dependencies` (it is currently present only transitively — promoting it to
a declared dependency avoids repeating the exact transitive-reliance mistake behind #322). Conservative
floor, e.g. `packaging (>=23.2)`; the `Requirement` / `SpecifierSet` / `canonicalize_name` /
`Marker.evaluate` APIs used here have been stable for years. The exact floor will be confirmed during
implementation.

### Call-site migration

All ~20 sites collapse to passing just the extra name. Each is mapped precisely during implementation,
preserving current intent. Representative cases:

| File | Before | After |
| --- | --- | --- |
| `modules/scoring/_bert.py` | `require("transformers", "transformers")` | `require("transformers")` |
| `modules/scoring/_ptuning/ptuning.py`, `_lora/lora.py` | `require("peft", extra="peft")` | `require("peft")` |
| `modules/scoring/_catboost/catboost_scorer.py` | `require("catboost", extra="catboost")` | `require("catboost")` |
| `_wrappers/embedder/vllm.py` | `require("vllm", extra="vllm")` | `require("vllm")` |
| `_wrappers/embedder/openai.py` | `require("tiktoken","openai")` + `require("openai","openai")` | `require("openai")` |
| `generation/_generator.py` | `require("openai","openai")` | `require("openai")` |
| `_wrappers/ranker.py` | `require("sentence_transformers", extra="sentence-transformers")` | `require("sentence-transformers")` |
| `_wrappers/embedder/sentence_transformers.py` | `require("transformers")` + `require("accelerate")` + `require("sentence_transformers")` | collapse (accelerate covered by recursion) |
| `_dump_tools/unit_dumpers.py` | peft / transformers / catboost requires | `require("peft")` / `require("transformers")` / `require("catboost")` |

### Testing (TDD, synthetic metadata)

Unit tests inject a fake requirement graph (monkeypatch `_iter_extra_reqs` or the underlying
`importlib.metadata.requires` / `version`) so assertions do not depend on what CI happens to install:

- all dependencies present and version-satisfied → no raise
- a leaf dependency missing → `ImportError` naming the dist and the extra + install hint
- installed version too old → `ImportError` showing installed vs required
- **nested-extra dependency missing (the `accelerate` case) → `ImportError` names `accelerate`** (proves #322)
- cyclic extras → resolution terminates
- unknown extra name → clear error

Plus one real-metadata smoke test against a known-present extra to catch wiring regressions.

Per project convention, tests are verified via CI (push branch + coverage dispatch), not run locally.

## Out of scope

- Changing what each module actually imports at runtime.
- Resolving full transitive *non-extra* dependency trees — we only walk declared extras, which is where
  the relevant guards belong.
- Source checkouts with no installed metadata (dev installs always have `.dist-info`/PEP 660 metadata).

## Rejected alternatives

- **Import-based presence check** — slower, drags in heavy imports (torch), and needs an
  import-name ↔ dist-name map.
- **One-level resolution (autointent extras only)** — would not catch `accelerate`, since it is nested
  inside `transformers`' `torch` extra; #322 would still need a manual guard.
- **Hand-maintained extra → deps map** — still a sync burden, just centralized.
