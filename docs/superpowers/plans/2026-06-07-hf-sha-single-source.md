# HF SHA Single Source of Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `DEFAULT_REVISIONS` the only place a HuggingFace commit SHA is written; eliminate ~13 duplicate SHA strings scattered across CI configs and tests.

**Architecture:** Extract `DEFAULT_REVISIONS` to a zero-dependency leaf module so `.ci/warm_hf_cache.py` can import it via a small `sys.path` shim without installing the autointent package. CI prewarm YAMLs become repo-IDs-only and the warm script joins repo IDs to SHAs at runtime. Tests and asset YAMLs drop redundant `revision=` lines and let the existing `HFModelConfig` validator fill them. A cross-check test asserts the prewarm subset is a subset of `DEFAULT_REVISIONS`.

**Tech Stack:** Python 3.10+, pydantic v2 (existing `HFModelConfig`), PyYAML (existing in warm script), pytest. No new dependencies.

**Branch:** `b/hf-rate-limit-on-tests` (continuing the in-flight refactor — user explicitly requested same branch).

**Spec:** [`docs/superpowers/specs/2026-06-07-hf-sha-single-source-design.md`](../specs/2026-06-07-hf-sha-single-source-design.md)

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/autointent/configs/_pinned_revisions.py` | **Create** | Leaf module holding `DEFAULT_REVISIONS` dict. Zero non-`__future__` imports. |
| `src/autointent/configs/_transformers.py` | Modify | Drop the literal dict; re-export `DEFAULT_REVISIONS` from the leaf module so all existing consumers keep working. |
| `.ci/warm_hf_cache.py` | Modify | Add `sys.path` shim to import the leaf module. Replace `_parse_entry` (split `repo@sha`) with `_resolve_entry` (look up SHA from `DEFAULT_REVISIONS`). |
| `.ci/hf-prewarm-linux.yaml` | Modify | Replace `repo@sha` entries with bare repo IDs. |
| `.ci/hf-prewarm-windows.yaml` | Modify | Same as linux. |
| `tests/ci/test_warm_hf_cache.py` | Modify | Delete `TestParseEntry` (parser no longer parses SHAs). Update `TestLoadConfig` fixtures to use bare repo IDs. Add `TestResolveEntry`. Add `test_prewarm_yamls_are_subset_of_default_revisions`. |
| `tests/modules/scoring/test_bert.py` | Modify | Drop the redundant `revision="..."` from the module-level `HFModelConfig(...)`. |
| `tests/modules/scoring/test_lora.py` | Modify | Same. |
| `tests/modules/scoring/test_ptuning.py` | Modify | Same. |
| `tests/assets/configs/multiclass.yaml` | Modify | Drop two `revision: ...` lines under catboost and ptuning embedder_config / classification_model_config. |
| `tests/assets/configs/multilabel.yaml` | Modify | Same — two `revision: ...` lines under catboost and ptuning. |
| `tests/configs/test_combined_config.py` | Modify | Add a structural test asserting the leaf module has no non-`__future__` imports. |

No other files change. Public API of `HFModelConfig` / `EmbedderConfig` / `CrossEncoderConfig` is preserved verbatim.

---

## Task 1: Extract DEFAULT_REVISIONS to a zero-dep leaf module

**Files:**
- Create: `src/autointent/configs/_pinned_revisions.py`
- Modify: `src/autointent/configs/_transformers.py` (lines 15–37)
- Test: `tests/configs/test_combined_config.py`

- [ ] **Step 1: Write the structural test that locks in the zero-import invariant**

Open `tests/configs/test_combined_config.py` and add this test at the bottom of the file:

```python
def test_pinned_revisions_module_has_no_runtime_imports():
    """The leaf module must be loadable without autointent's deps installed.

    .ci/warm_hf_cache.py imports it via a sys.path shim that does NOT
    install pydantic, datasets, or any other autointent dep. If a future
    edit adds e.g. `import json` to _pinned_revisions, the warm-cache job
    silently keeps working in environments that happen to have json
    available but breaks in stricter ones; this test prevents that drift.
    """
    import ast
    from pathlib import Path

    from autointent.configs import _pinned_revisions

    source = Path(_pinned_revisions.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module == "__future__", (
                f"_pinned_revisions.py must not import {node.module!r} "
                f"(only `from __future__ import ...` is allowed)"
            )
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
            raise AssertionError(
                f"_pinned_revisions.py must not contain `import` statements; "
                f"found: {modules}"
            )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/configs/test_combined_config.py::test_pinned_revisions_module_has_no_runtime_imports -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autointent.configs._pinned_revisions'`

- [ ] **Step 3: Create the leaf module**

Create `src/autointent/configs/_pinned_revisions.py` with exactly this content:

```python
"""Pinned commit SHAs for Hugging Face models used as defaults in autointent.

This module is the canonical source of truth for every SHA pin in the
project. The dict is consumed by:

  - autointent.configs._transformers.HFModelConfig._apply_default_revision,
    which auto-fills `revision` on configs whose model_name is a key here
  - .ci/warm_hf_cache.py, which joins repo IDs from the prewarm YAML
    against this dict to produce pinned entries for the cache warmer

LEAF MODULE INVARIANT: this file must contain only `from __future__`
imports (no other imports of any kind). It is loaded by warm_hf_cache.py
via a sys.path shim in an environment where the autointent package is
NOT installed; any non-__future__ import will break that consumer.

A unit test (tests/configs/test_combined_config.py::
test_pinned_revisions_module_has_no_runtime_imports) enforces this
invariant via ast parsing. Do not relax the test to add an import.

Update an entry below when you intentionally want to move a default to a
newer revision. To add a new pinned model, add a new entry here, then
(optionally) list its repo ID in .ci/hf-prewarm-linux.yaml /
hf-prewarm-windows.yaml if it should be CI-prewarmed.
"""

from __future__ import annotations

DEFAULT_REVISIONS: dict[str, str] = {
    "prajjwal1/bert-tiny": "79779625a0a40f1eee8496e16056bc0d7766df22",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "intfloat/multilingual-e5-large-instruct": "274baa43b0e13e37fafa6428dbc7938e62e5c439",
    "intfloat/multilingual-e5-small": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
    "avsolatorio/GIST-small-Embedding-v0": "75e62fd210b9fde790430e0b2f040b0b00a021b1",
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    # Used heavily in the embedder test suite (tests/embedder/conftest.py).
    # Pinning the SHA here lets HFModelConfig auto-fill it via the validator
    # so sentence-transformers never asks the Hub for "main" — which 429s
    # under parallel matrix load even when the model files are cached.
    "sergeyzh/rubert-tiny-turbo": "93769a3baad2b037e5c2e4312fccf6bcfe082bf1",
    "microsoft/deberta-v3-large": "64a8c8eab3e352a784c658aef62be1662607476f",
    "microsoft/deberta-v3-small": "a36c739020e01763fe789b4b85e2df55d6180012",
}
```

- [ ] **Step 4: Replace the dict in `_transformers.py` with a re-export**

Open `src/autointent/configs/_transformers.py`. Replace lines 15–37 (the comment block + the `DEFAULT_REVISIONS: dict[str, str] = { ... }` literal) with a single re-export. The exact edit:

Find this block (lines 15–37):

```python
# Pinned commit SHAs for the Hugging Face models that ship as defaults in
# autointent. When a config is constructed with one of these model_name values
# and no explicit ``revision``, the SHA below is filled in automatically so the
# library never has to call the HF API just to resolve ``main`` to a hash for
# cache keying. Update an entry here when you intentionally want to move a
# default to a newer revision.
DEFAULT_REVISIONS: dict[str, str] = {
    "prajjwal1/bert-tiny": "79779625a0a40f1eee8496e16056bc0d7766df22",
    # ... 10 more entries ...
    "microsoft/deberta-v3-small": "a36c739020e01763fe789b4b85e2df55d6180012",
}
```

Replace with:

```python
# DEFAULT_REVISIONS is the canonical source of truth for every pinned
# HuggingFace commit SHA in the project. It lives in a zero-dep leaf
# module so .ci/warm_hf_cache.py can import it without installing the
# autointent package. See src/autointent/configs/_pinned_revisions.py.
from autointent.configs._pinned_revisions import DEFAULT_REVISIONS  # noqa: F401, E402
```

Place this immediately after the existing `if TYPE_CHECKING: ...` block (around line 12). The `# noqa: F401` is needed because the re-export looks unused inside this file; the `# noqa: E402` is needed because the import sits below the `if TYPE_CHECKING` block per the existing file structure.

- [ ] **Step 5: Run the leaf-module test to verify it passes**

Run: `uv run pytest tests/configs/test_combined_config.py::test_pinned_revisions_module_has_no_runtime_imports -v`
Expected: PASS

- [ ] **Step 6: Run the full configs test module to verify the re-export works**

Run: `uv run pytest tests/configs/ -v`
Expected: all tests PASS (no regressions in the existing DEFAULT_REVISIONS-using tests at `test_combined_config.py::test_deberta_v3_large_is_pinned`, `::test_deberta_v3_small_is_pinned`, `::test_canonical_test_models_have_pinned_revisions`).

- [ ] **Step 7: Quick smoke check that the validator path still works**

Run: `uv run python -c "from autointent.configs import HFModelConfig; c = HFModelConfig(model_name='prajjwal1/bert-tiny'); print(c.revision)"`
Expected output: `79779625a0a40f1eee8496e16056bc0d7766df22`

- [ ] **Step 8: Commit**

```bash
git add src/autointent/configs/_pinned_revisions.py src/autointent/configs/_transformers.py tests/configs/test_combined_config.py
git commit -m "refactor(configs): extract DEFAULT_REVISIONS to leaf module for SHA single-source

The pin dict is now in src/autointent/configs/_pinned_revisions.py
(zero non-__future__ imports) so .ci/warm_hf_cache.py can import it via
a sys.path shim without installing the autointent package.

_transformers.py re-exports the symbol; all existing consumers keep
working unchanged. A new structural test enforces the zero-import
invariant via ast parsing."
```

---

## Task 2: Rewrite warm script to resolve SHAs via DEFAULT_REVISIONS; flip prewarm YAMLs to repo-IDs-only

**Files:**
- Modify: `.ci/warm_hf_cache.py` (add sys.path shim; replace `_parse_entry` with `_resolve_entry`; update `_load_config`)
- Modify: `.ci/hf-prewarm-linux.yaml`
- Modify: `.ci/hf-prewarm-windows.yaml`
- Modify: `tests/ci/test_warm_hf_cache.py` (delete `TestParseEntry`; rewrite `TestLoadConfig` fixtures; add `TestResolveEntry`)

- [ ] **Step 1: Rewrite the test file with the new contract**

The current test file references `wc.parse_entry` and `wc.load_config` (no underscores) which don't exist on the module — pre-existing breakage from commit `ec4cd5c6` that this task naturally fixes. Replace the **entire** contents of `tests/ci/test_warm_hf_cache.py` with:

```python
from __future__ import annotations

import pytest
import warm_hf_cache as wc


class TestResolveEntry:
    def test_known_repo_resolves_to_pinned_sha(self):
        entry = wc._resolve_entry("prajjwal1/bert-tiny", "model")
        assert entry.repo_id == "prajjwal1/bert-tiny"
        assert entry.repo_type == "model"
        # The SHA must match DEFAULT_REVISIONS — we look it up here rather
        # than hardcoding to keep this test honest if the pin moves.
        from autointent.configs._pinned_revisions import DEFAULT_REVISIONS
        assert entry.revision == DEFAULT_REVISIONS["prajjwal1/bert-tiny"]

    def test_unknown_repo_raises(self):
        with pytest.raises(wc.ConfigError, match="not in DEFAULT_REVISIONS"):
            wc._resolve_entry("not-a-real/repo", "model")


class TestLoadConfig:
    def test_empty_file(self, tmp_path):
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("models: []\ndatasets: []\n")
        entries = wc._load_config(cfg)
        assert entries == []

    def test_models_resolved_via_default_revisions(self, tmp_path):
        from autointent.configs._pinned_revisions import DEFAULT_REVISIONS

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "models:\n"
            "  - prajjwal1/bert-tiny\n"
            "datasets: []\n"
        )
        entries = wc._load_config(cfg)
        assert entries == [
            wc.Entry(
                repo_type="model",
                repo_id="prajjwal1/bert-tiny",
                revision=DEFAULT_REVISIONS["prajjwal1/bert-tiny"],
            ),
        ]

    def test_unknown_top_level_key_raises(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("models: []\nfoo: []\n")
        with pytest.raises(wc.ConfigError, match="Unknown top-level"):
            wc._load_config(cfg)

    def test_unpinned_model_raises(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("models:\n  - not-a-real/repo\n")
        with pytest.raises(wc.ConfigError, match="not in DEFAULT_REVISIONS"):
            wc._load_config(cfg)

    def test_dataset_entry_raises(self, tmp_path):
        # DEFAULT_REVISIONS covers models only today. If we ever need to
        # warm a dataset, _resolve_entry must be extended; the parser
        # raises until then so a dataset in the YAML can't silently
        # regress to an unpinned download.
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "models: []\n"
            "datasets:\n"
            "  - DeepPavlov/clinc150\n"
        )
        with pytest.raises(wc.ConfigError, match="not in DEFAULT_REVISIONS"):
            wc._load_config(cfg)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/ci/test_warm_hf_cache.py -v`
Expected: FAIL — `AttributeError: module 'warm_hf_cache' has no attribute '_resolve_entry'`

- [ ] **Step 3: Modify `.ci/warm_hf_cache.py` to add the sys.path shim and replace the parser**

Open `.ci/warm_hf_cache.py`. Make three edits:

**Edit 3a** — Add the sys.path shim. Find this block at lines 22–35:

```python
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
```

Replace with:

```python
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

# sys.path shim: load DEFAULT_REVISIONS from the autointent leaf module
# without installing the package. The leaf module is by contract
# zero-dependency (enforced by
# tests/configs/test_combined_config.py::test_pinned_revisions_module_has_no_runtime_imports)
# so this import succeeds in the warm-cache job's slim environment.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from autointent.configs._pinned_revisions import DEFAULT_REVISIONS  # noqa: E402
```

Note: `re` is no longer imported because `_SHA_RE` is gone (SHAs are no longer parsed from text).

**Edit 3b** — Replace `_parse_entry` with `_resolve_entry`. Find this block at lines 102–116:

```python
def _parse_entry(text: str) -> tuple[str, str]:
    """Split ``"<repo>@<sha>"`` into ``(repo, sha)`` and validate the SHA shape.

    Raises:
        ConfigError: if no ``@`` is present, or if the revision is not a
            40-char lowercase hex string.
    """
    if "@" not in text:
        msg = f"Entry {text!r} must be pinned to a SHA (use 'repo@<40-char hex>')"
        raise ConfigError(msg)
    repo, _, rev = text.partition("@")
    if not _SHA_RE.fullmatch(rev):
        msg = f"Entry {text!r}: revision {rev!r} is not a 40-char hex SHA"
        raise ConfigError(msg)
    return repo, rev
```

Replace with:

```python
def _resolve_entry(repo_id: str, repo_type: str) -> Entry:
    """Resolve a repo ID to a pinned Entry via DEFAULT_REVISIONS.

    Raises:
        ConfigError: if ``repo_id`` is not pinned in DEFAULT_REVISIONS.
            DEFAULT_REVISIONS covers models only today, so any
            ``repo_type="dataset"`` entry will raise here until the dict
            is extended; that's intentional (a dataset in the YAML must
            not silently regress to an unpinned download).
    """
    if repo_id not in DEFAULT_REVISIONS:
        msg = (
            f"{repo_id!r} ({repo_type}) not in DEFAULT_REVISIONS. Add a pin "
            "to src/autointent/configs/_pinned_revisions.py before listing "
            "the repo in .ci/hf-prewarm-*.yaml."
        )
        raise ConfigError(msg)
    return Entry(repo_type=repo_type, repo_id=repo_id, revision=DEFAULT_REVISIONS[repo_id])
```

**Edit 3c** — Update `_load_config` to use `_resolve_entry`. Find this block at lines 119–136:

```python
def _load_config(path: Path) -> list[Entry]:
    """Load ``hf-prewarm.yaml`` and return a flat list of entries.

    Raises:
        ConfigError: on unknown top-level keys or malformed entries.
    """
    data = yaml.safe_load(path.read_text()) or {}
    known = {"models", "datasets"}
    unknown = set(data) - known
    if unknown:
        msg = f"Unknown top-level keys in {path}: {sorted(unknown)}"
        raise ConfigError(msg)
    entries: list[Entry] = []
    for key, repo_type in (("models", "model"), ("datasets", "dataset")):
        for raw in data.get(key, []) or []:
            repo, rev = _parse_entry(raw)
            entries.append(Entry(repo_type=repo_type, repo_id=repo, revision=rev))
    return entries
```

Replace with:

```python
def _load_config(path: Path) -> list[Entry]:
    """Load ``hf-prewarm.yaml`` and return a flat list of entries.

    Entries in the YAML are bare repo IDs (no ``@sha`` suffix); SHAs are
    looked up from DEFAULT_REVISIONS in _resolve_entry.

    Raises:
        ConfigError: on unknown top-level keys, or on a repo_id that's
            not pinned in DEFAULT_REVISIONS.
    """
    data = yaml.safe_load(path.read_text()) or {}
    known = {"models", "datasets"}
    unknown = set(data) - known
    if unknown:
        msg = f"Unknown top-level keys in {path}: {sorted(unknown)}"
        raise ConfigError(msg)
    entries: list[Entry] = []
    for key, repo_type in (("models", "model"), ("datasets", "dataset")):
        for repo_id in data.get(key, []) or []:
            entries.append(_resolve_entry(repo_id, repo_type))
    return entries
```

- [ ] **Step 4: Run the test file to verify it passes**

Run: `uv run pytest tests/ci/test_warm_hf_cache.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Flip both prewarm YAMLs to repo-IDs-only**

Replace the entire contents of `.ci/hf-prewarm-linux.yaml` with:

```yaml
# Canonical test-model set. See docs/superpowers/specs/2026-06-06-hf-test-refactor-design.md.
# Every CI test that genuinely needs an HF model uses one of these three;
# every other test path either uses HashingVectorizerEmbeddingConfig (via
# tests/conftest.py::get_test_embedder_config) or loads from a local asset.
#
# Entries are BARE repo IDs. SHAs are looked up from
# src/autointent/configs/_pinned_revisions.py::DEFAULT_REVISIONS by
# .ci/warm_hf_cache.py at runtime. Adding a repo here without adding a
# matching DEFAULT_REVISIONS entry is caught two ways:
#   1. The cross-check test in tests/ci/test_warm_hf_cache.py
#   2. The warm-cache job itself raises ConfigError on unknown repo
#
# tests/conftest.py also runs a session-scoped autouse guard that fails
# any huggingface_hub call whose revision is not a 40-hex SHA.
models:
  - prajjwal1/bert-tiny
  - cross-encoder/ms-marco-MiniLM-L6-v2
  - sergeyzh/rubert-tiny-turbo
datasets: []
```

Replace the entire contents of `.ci/hf-prewarm-windows.yaml` with:

```yaml
# Canonical test-model set. See docs/superpowers/specs/2026-06-06-hf-test-refactor-design.md.
# Identical content to .ci/hf-prewarm-linux.yaml after the test refactor:
# the per-OS split is retained for cache-key namespacing but the model
# list happens to be the same on both OSes. Format is bare repo IDs; SHAs
# are joined from DEFAULT_REVISIONS by .ci/warm_hf_cache.py at runtime.
models:
  - prajjwal1/bert-tiny
  - cross-encoder/ms-marco-MiniLM-L6-v2
  - sergeyzh/rubert-tiny-turbo
datasets: []
```

- [ ] **Step 6: Smoke-test the warm script end-to-end with the new YAML**

Run: `uv run --with 'huggingface_hub[hf_xet]' --with pyyaml --with datasets python .ci/warm_hf_cache.py --config .ci/hf-prewarm-linux.yaml`
Expected output ends with a Summary line indicating 3 cached or downloaded, 0 failed. (If your local cache is cold, downloads will happen — that's fine.)

- [ ] **Step 7: Verify the unit-tests path still collects cleanly**

Run: `uv run pytest tests/ci/ -v`
Expected: all tests PASS (the only test file in that directory is the one rewritten in Step 1).

- [ ] **Step 8: Commit**

```bash
git add .ci/warm_hf_cache.py .ci/hf-prewarm-linux.yaml .ci/hf-prewarm-windows.yaml tests/ci/test_warm_hf_cache.py
git commit -m "refactor(ci): warm_hf_cache resolves SHAs via DEFAULT_REVISIONS

Prewarm YAMLs now carry bare repo IDs; the warm script joins them to
SHAs from DEFAULT_REVISIONS at runtime via a sys.path shim that imports
the zero-dep _pinned_revisions leaf module without installing autointent.

Also fixes pre-existing breakage in tests/ci/test_warm_hf_cache.py where
the test file referenced public 'parse_entry'/'load_config' that didn't
exist on the module (commit ec4cd5c6 renamed them to underscore-prefixed
without updating the tests)."
```

---

## Task 3: Drop redundant `revision=` from scoring tests

**Files:**
- Modify: `tests/modules/scoring/test_bert.py:13`
- Modify: `tests/modules/scoring/test_lora.py:15`
- Modify: `tests/modules/scoring/test_ptuning.py:15`

This task is a behavior no-op: `HFModelConfig._apply_default_revision` already fills `revision` when it's omitted, so the deleted lines produce an identical config. We verify with the existing test suite.

- [ ] **Step 1: Edit `tests/modules/scoring/test_bert.py`**

Find line 13:

```python
_config = HFModelConfig(model_name="prajjwal1/bert-tiny", revision="79779625a0a40f1eee8496e16056bc0d7766df22")
```

Replace with:

```python
_config = HFModelConfig(model_name="prajjwal1/bert-tiny")
```

- [ ] **Step 2: Edit `tests/modules/scoring/test_lora.py`**

Find line 15:

```python
_config = HFModelConfig(model_name="prajjwal1/bert-tiny", revision="79779625a0a40f1eee8496e16056bc0d7766df22")
```

Replace with:

```python
_config = HFModelConfig(model_name="prajjwal1/bert-tiny")
```

- [ ] **Step 3: Edit `tests/modules/scoring/test_ptuning.py`**

Find line 15:

```python
_config = HFModelConfig(model_name="prajjwal1/bert-tiny", revision="79779625a0a40f1eee8496e16056bc0d7766df22")
```

Replace with:

```python
_config = HFModelConfig(model_name="prajjwal1/bert-tiny")
```

- [ ] **Step 4: Add a focused assertion that the validator filled `revision` correctly**

We need to lock in the no-op claim with one test, otherwise a future change to `_apply_default_revision` could silently break these test fixtures. Add to `tests/configs/test_combined_config.py`, near the existing `test_canonical_test_models_have_pinned_revisions`:

```python
def test_bert_tiny_config_in_scoring_tests_gets_pinned_revision():
    """test_bert/lora/ptuning.py rely on _apply_default_revision to fill
    revision when they omit it. Lock that contract in here so a future
    change to the validator doesn't silently make the scoring tests
    contact HF Hub for revision resolution."""
    from autointent.configs import HFModelConfig
    from autointent.configs._pinned_revisions import DEFAULT_REVISIONS

    cfg = HFModelConfig(model_name="prajjwal1/bert-tiny")
    assert cfg.revision == DEFAULT_REVISIONS["prajjwal1/bert-tiny"]
```

- [ ] **Step 5: Run the new assertion to verify the validator path**

Run: `uv run pytest tests/configs/test_combined_config.py::test_bert_tiny_config_in_scoring_tests_gets_pinned_revision -v`
Expected: PASS

- [ ] **Step 6: Run the three scoring test files to verify no regression**

These tests require the `transformers` extra. Run with it explicitly:

Run: `uv run --extra transformers pytest tests/modules/scoring/test_bert.py tests/modules/scoring/test_lora.py tests/modules/scoring/test_ptuning.py -v`

Expected: all tests PASS (or SKIP if `accelerate`/`peft` are not present — that's expected and matches the prior behavior; what matters is no new FAILED).

- [ ] **Step 7: Commit**

```bash
git add tests/modules/scoring/test_bert.py tests/modules/scoring/test_lora.py tests/modules/scoring/test_ptuning.py tests/configs/test_combined_config.py
git commit -m "test(scoring): drop redundant revision= from bert/lora/ptuning HFModelConfig

HFModelConfig._apply_default_revision auto-fills revision from
DEFAULT_REVISIONS, so the explicit pin in the test fixtures was
duplicated. Add an assertion in test_combined_config.py to lock in the
validator behavior these tests now depend on."
```

---

## Task 4: Drop redundant `revision:` from search-space YAML configs

**Files:**
- Modify: `tests/assets/configs/multiclass.yaml:42, 67`
- Modify: `tests/assets/configs/multilabel.yaml:38, 55`

Same no-op claim as Task 3 — the YAML is loaded into `HFModelConfig`, which fills the revision. Plus `tests/conftest.py::_rewrite_field` already drops `revision` when retargeting `model_name` for preset tests, so the YAML's `revision:` line is fully redundant.

- [ ] **Step 1: Edit `tests/assets/configs/multiclass.yaml`**

Find this block (lines 40–42, inside the catboost module entry):

```yaml
      embedder_config:
        - model_name: prajjwal1/bert-tiny
          revision: 79779625a0a40f1eee8496e16056bc0d7766df22
```

Replace with:

```yaml
      embedder_config:
        - model_name: prajjwal1/bert-tiny
```

Find this block (lines 65–67, inside the ptuning module entry):

```yaml
      classification_model_config:
        - model_name: "prajjwal1/bert-tiny"
          revision: 79779625a0a40f1eee8496e16056bc0d7766df22
```

Replace with:

```yaml
      classification_model_config:
        - model_name: "prajjwal1/bert-tiny"
```

- [ ] **Step 2: Edit `tests/assets/configs/multilabel.yaml`**

Find this block (lines 35–38, inside the catboost module entry):

```yaml
      embedder_config:
        - null
        - model_name: prajjwal1/bert-tiny
          revision: 79779625a0a40f1eee8496e16056bc0d7766df22
```

Replace with:

```yaml
      embedder_config:
        - null
        - model_name: prajjwal1/bert-tiny
```

Find this block (lines 53–55, inside the ptuning module entry):

```yaml
      classification_model_config:
        - model_name: prajjwal1/bert-tiny
          revision: 79779625a0a40f1eee8496e16056bc0d7766df22
```

Replace with:

```yaml
      classification_model_config:
        - model_name: prajjwal1/bert-tiny
```

- [ ] **Step 3: Confirm no other `revision:` lines remain in either file**

Run: `grep -n 'revision' tests/assets/configs/multiclass.yaml tests/assets/configs/multilabel.yaml`
Expected: empty output.

- [ ] **Step 4: Run preset/optimization/inference tests to confirm no regression**

These are the consumers of multiclass.yaml / multilabel.yaml. Run with the full extras matrix used in CI:

Run: `uv run --extra catboost --extra peft --extra transformers --extra sentence-transformers pytest tests/pipeline/ -v -x`

Expected: all tests PASS (or appropriately SKIP based on missing optional extras). What matters: no new FAILED, and in particular no `Unpinned HF call` assertion errors from the conftest guard.

- [ ] **Step 5: Commit**

```bash
git add tests/assets/configs/multiclass.yaml tests/assets/configs/multilabel.yaml
git commit -m "test(configs): drop redundant revision: from multiclass/multilabel search-space YAMLs

HFModelConfig._apply_default_revision fills revision when omitted.
tests/conftest.py::_rewrite_field also drops revision when retargeting
model_name in preset tests, so the YAML pin was duplicated work."
```

---

## Task 5: Add cross-check test asserting prewarm YAMLs are a subset of DEFAULT_REVISIONS

**Files:**
- Modify: `tests/ci/test_warm_hf_cache.py`

This is a defense-in-depth test that catches drift: if someone adds `BAAI/bge-something@<sha>` to a prewarm YAML in the old format, or adds a bare repo ID that's not in `DEFAULT_REVISIONS`, the test catches it before the warm-cache job runs.

- [ ] **Step 1: Write the cross-check test**

Append this test class to `tests/ci/test_warm_hf_cache.py`:

```python
class TestPrewarmConfigsAreSubsetOfDefaultRevisions:
    """Defense-in-depth: the warm-cache job itself raises ConfigError on
    unknown repo IDs (via _resolve_entry), but that error only fires at
    CI time. This test catches the same drift at unit-test time so a
    misconfigured YAML never reaches the warm-cache job."""

    @pytest.mark.parametrize(
        "yaml_path",
        [".ci/hf-prewarm-linux.yaml", ".ci/hf-prewarm-windows.yaml"],
    )
    def test_every_model_is_pinned_in_default_revisions(self, yaml_path):
        import yaml as pyyaml
        from pathlib import Path

        from autointent.configs._pinned_revisions import DEFAULT_REVISIONS

        repo_root = Path(__file__).resolve().parents[2]
        data = pyyaml.safe_load((repo_root / yaml_path).read_text()) or {}
        models = data.get("models") or []
        missing = [m for m in models if m not in DEFAULT_REVISIONS]
        assert not missing, (
            f"{yaml_path}: {missing} not in DEFAULT_REVISIONS. Add a pin to "
            f"src/autointent/configs/_pinned_revisions.py."
        )

    @pytest.mark.parametrize(
        "yaml_path",
        [".ci/hf-prewarm-linux.yaml", ".ci/hf-prewarm-windows.yaml"],
    )
    def test_no_sha_suffix_in_repo_ids(self, yaml_path):
        """The new YAML format is bare repo IDs. A '@' in an entry means
        someone added an entry in the old 'repo@sha' format — likely
        because they copy-pasted from git history. Catch it explicitly so
        the error message points at the right fix."""
        import yaml as pyyaml
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        data = pyyaml.safe_load((repo_root / yaml_path).read_text()) or {}
        with_sha = [m for m in (data.get("models") or []) if "@" in m]
        assert not with_sha, (
            f"{yaml_path}: {with_sha} use the old 'repo@sha' format. "
            "Drop the '@<sha>' suffix; SHAs are looked up from "
            "DEFAULT_REVISIONS by warm_hf_cache.py."
        )
```

- [ ] **Step 2: Run the new tests to verify they pass against the current state**

Run: `uv run pytest tests/ci/test_warm_hf_cache.py::TestPrewarmConfigsAreSubsetOfDefaultRevisions -v`
Expected: all 4 parametrized tests PASS (2 files × 2 tests).

- [ ] **Step 3: Verify the test actually catches drift (manual sanity check)**

Temporarily edit `.ci/hf-prewarm-linux.yaml` to add a fake entry:

```yaml
models:
  - prajjwal1/bert-tiny
  - cross-encoder/ms-marco-MiniLM-L6-v2
  - sergeyzh/rubert-tiny-turbo
  - fake/not-pinned-anywhere
datasets: []
```

Run: `uv run pytest tests/ci/test_warm_hf_cache.py::TestPrewarmConfigsAreSubsetOfDefaultRevisions -v`
Expected: `test_every_model_is_pinned_in_default_revisions[.ci/hf-prewarm-linux.yaml]` FAILS with a message containing `['fake/not-pinned-anywhere'] not in DEFAULT_REVISIONS`.

Revert the temporary edit. Re-run to confirm green:

Run: `git checkout .ci/hf-prewarm-linux.yaml`
Run: `uv run pytest tests/ci/test_warm_hf_cache.py::TestPrewarmConfigsAreSubsetOfDefaultRevisions -v`
Expected: all PASS.

- [ ] **Step 4: Run the full warm-cache test file as a final regression check**

Run: `uv run pytest tests/ci/test_warm_hf_cache.py -v`
Expected: all tests PASS (6 from Task 2 + 4 new = 10 total).

- [ ] **Step 5: Run the acceptance grep from spec §8**

Run: `grep -rn -E '[0-9a-f]{40}' src/ tests/ .ci/ --include='*.py' --include='*.yaml' --include='*.yml'`

Expected output contains literal SHAs only in:
- `src/autointent/configs/_pinned_revisions.py` (the source-of-truth dict)
- `tests/configs/test_combined_config.py` (reads DEFAULT_REVISIONS in assertions; should NOT contain raw SHA literals other than as values being looked up)
- `tests/ci/test_warm_hf_cache.py` (only if it reads DEFAULT_REVISIONS — should not contain raw SHA literals after Task 2)
- Optionally `.ci/warm_hf_cache.py` if any docstring/comment shows an example SHA — review and remove if so.

NO matches in any `.yaml` file under `.ci/` or `tests/assets/`. If any unexpected literal SHA appears, that's a refactor gap — fix before commit.

- [ ] **Step 6: Run the full test suite as the final sign-off**

Run: `uv run --extra catboost --extra peft --extra transformers --extra sentence-transformers pytest -n auto`
Expected: same pass/skip/xfail counts as before this refactor started. In particular: zero new `Unpinned HF call` AssertionErrors.

- [ ] **Step 7: Commit**

```bash
git add tests/ci/test_warm_hf_cache.py
git commit -m "test(ci): assert prewarm YAML models are subset of DEFAULT_REVISIONS

Catches drift at unit-test time (before the warm-cache job runs):
- Every repo ID in .ci/hf-prewarm-*.yaml must be pinned in
  DEFAULT_REVISIONS
- No entry may use the old 'repo@sha' format (bare repo IDs only)

Together with the runtime guard in warm_hf_cache._resolve_entry, this
makes the SHA single-source invariant impossible to break silently."
```

---

## Self-review against the spec

**Spec coverage check:**

| Spec section | Task |
|---|---|
| §3.1 — Leaf module for the dict | Task 1 |
| §3.2 — Prewarm YAMLs lose SHAs | Task 2 (Step 5) |
| §3.3 — `warm_hf_cache.py` joins via DEFAULT_REVISIONS | Task 2 (Step 3) |
| §3.4 — Test/YAML cleanup | Tasks 3 + 4 |
| §3.5 — Cross-check test | Task 5 |
| §3.6 — `test_warm_hf_cache.py` updates | Task 2 (Step 1) |
| §4 — Migration safety (phase ordering, behavioral no-ops verified by tests) | Tasks 1–5 are independently green commits |
| §5 — Risk mitigation (zero-dep leaf module, re-export covers old import paths) | Task 1's structural test + Task 2's smoke run |
| §6 — Migration order | Tasks 1 → 2 → 3 → 4 → 5 (matches phases 1 → 2 → 3 → 4) |
| §8 — Acceptance grep | Task 5 (Step 5) |

No spec gaps. Tasks 3 and 4 together cover spec §6 Phase 3 (split for cleaner per-file verification).

**Placeholder scan:** no "TBD", no "implement later", no "similar to Task N", no steps without exact code or commands. Every commit message is fully written.

**Type/name consistency:**
- `DEFAULT_REVISIONS` referenced consistently (Tasks 1, 2, 3, 5).
- `_resolve_entry(repo_id: str, repo_type: str) -> Entry` signature consistent across Task 2 Steps 1 and 3.
- `_load_config(path: Path) -> list[Entry]` signature unchanged from current code.
- `Entry(repo_type=..., repo_id=..., revision=...)` field names match the existing dataclass (verified against `.ci/warm_hf_cache.py:48–54`).
- `src/autointent/configs/_pinned_revisions.py` path written identically every time.
