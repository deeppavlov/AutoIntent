# HF Cache Warmer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-workflow HF prewarm pattern with a single orchestrator workflow that warms every HF model/dataset the test suite needs, in one place, before any test job runs.

**Architecture:** A new `ci.yaml` workflow runs a `warm-cache` job (Linux + Windows matrix) that invokes `.ci/warm_hf_cache.py`. The script reads a declarative list of `repo@sha` entries from `.ci/hf-prewarm.yaml`, fast-paths cached snapshots with `local_files_only=True`, and downloads only what's missing with retries. All test jobs `needs: warm-cache` and inherit the warmed cache via restore-keys. The 6 standalone `test-*.yaml` files collapse into ci.yaml jobs that still `uses: ./.github/workflows/reusable-test.yaml`.

**Tech Stack:** GitHub Actions, Python (stdlib + huggingface_hub + PyYAML), uv for transient deps, actions/cache@v4.

**Spec:** `docs/superpowers/specs/2026-06-05-hf-cache-warmer-design.md`

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Create | `.ci/hf-prewarm.yaml` | Declarative `repo@sha` list; single source of truth |
| Create | `.ci/warm_hf_cache.py` | Read the list, fast-path cached, download missing with retries |
| Create | `tests/ci/__init__.py` | Make `tests/ci/` a package |
| Create | `tests/ci/conftest.py` | Put `.ci/` on `sys.path` so the script is importable |
| Create | `tests/ci/test_warm_hf_cache.py` | Pure-function unit tests for the script |
| Create | `.github/workflows/ci.yaml` | Orchestrator: warm-cache job + all test jobs |
| Modify | `.github/workflows/reusable-test.yaml` | Prefer warmed cache in restore-keys; remove inline prewarm step + `prewarm_models` input |
| Delete | `.github/workflows/unit-tests.yaml` | Folded into ci.yaml |
| Delete | `.github/workflows/test-embedder.yaml` | Folded into ci.yaml |
| Delete | `.github/workflows/test-scorers.yaml` | Folded into ci.yaml |
| Delete | `.github/workflows/test-presets.yaml` | Folded into ci.yaml |
| Delete | `.github/workflows/test-optimization.yaml` | Folded into ci.yaml |
| Delete | `.github/workflows/test-inference.yaml` | Folded into ci.yaml |

---

### Task 1: Declarative prewarm list

**Files:**
- Create: `.ci/hf-prewarm.yaml`

- [ ] **Step 1: Create the directory and file**

```bash
mkdir -p .ci
```

- [ ] **Step 2: Write `.ci/hf-prewarm.yaml`**

```yaml
# Single source of truth for every HF model and dataset that the autointent
# test suite needs to have on disk. Used by .ci/warm_hf_cache.py to populate
# the actions/cache@v4 HF cache before any test job runs.
#
# Format: "<repo_id>@<sha>". The SHA must be a 40-char hex commit — this is
# what lets the warm-cache fast path skip the HF API entirely (a revision
# string that looks like a SHA is locally verifiable without a model_info
# call). Adding an unpinned entry is a validation error in the script.
models:
  - prajjwal1/bert-tiny@79779625a0a40f1eee8496e16056bc0d7766df22
  - sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41
  - intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3
  - intfloat/multilingual-e5-large-instruct@274baa43b0e13e37fafa6428dbc7938e62e5c439
  - cross-encoder/ms-marco-MiniLM-L6-v2@c5ee24cb16019beea0893ab7796b1df96625c6b8
  - avsolatorio/GIST-small-Embedding-v0@75e62fd210b9fde790430e0b2f040b0b00a021b1
  - BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a
  - BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
  - sergeyzh/rubert-tiny-turbo@93769a3baad2b037e5c2e4312fccf6bcfe082bf1
  - microsoft/deberta-v3-small@a36c739020e01763fe789b4b85e2df55d6180012
  - microsoft/deberta-v3-large@64a8c8eab3e352a784c658aef62be1662607476f
datasets:
  - DeepPavlov/clinc150@d835118ecd5ffe5488d22e9e58d1c23d18c33229
```

- [ ] **Step 3: Commit**

```bash
git add .ci/hf-prewarm.yaml
git commit -m "ci: add declarative HF prewarm list"
```

---

### Task 2: Warmer script — pure helpers + tests

**Files:**
- Create: `.ci/warm_hf_cache.py` (skeleton with pure helpers only — download logic comes in Task 3)
- Create: `tests/ci/__init__.py`
- Create: `tests/ci/conftest.py`
- Create: `tests/ci/test_warm_hf_cache.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ci/test_warm_hf_cache.py
from __future__ import annotations

import pytest

import warm_hf_cache as wc


class TestParseEntry:
    def test_valid_sha(self):
        repo, rev = wc.parse_entry("prajjwal1/bert-tiny@79779625a0a40f1eee8496e16056bc0d7766df22")
        assert repo == "prajjwal1/bert-tiny"
        assert rev == "79779625a0a40f1eee8496e16056bc0d7766df22"

    def test_missing_sha_raises(self):
        with pytest.raises(wc.ConfigError, match="must be pinned"):
            wc.parse_entry("prajjwal1/bert-tiny")

    def test_non_hex_revision_raises(self):
        with pytest.raises(wc.ConfigError, match="40-char hex"):
            wc.parse_entry("prajjwal1/bert-tiny@main")

    def test_short_revision_raises(self):
        with pytest.raises(wc.ConfigError, match="40-char hex"):
            wc.parse_entry("prajjwal1/bert-tiny@deadbeef")


class TestLoadConfig:
    def test_empty_file(self, tmp_path):
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("models: []\ndatasets: []\n")
        entries = wc.load_config(cfg)
        assert entries == []

    def test_models_and_datasets(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "models:\n"
            "  - prajjwal1/bert-tiny@79779625a0a40f1eee8496e16056bc0d7766df22\n"
            "datasets:\n"
            "  - DeepPavlov/clinc150@d835118ecd5ffe5488d22e9e58d1c23d18c33229\n"
        )
        entries = wc.load_config(cfg)
        assert entries == [
            wc.Entry(repo_type="model", repo_id="prajjwal1/bert-tiny",
                     revision="79779625a0a40f1eee8496e16056bc0d7766df22"),
            wc.Entry(repo_type="dataset", repo_id="DeepPavlov/clinc150",
                     revision="d835118ecd5ffe5488d22e9e58d1c23d18c33229"),
        ]

    def test_unknown_top_level_key_raises(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("models: []\nfoo: []\n")
        with pytest.raises(wc.ConfigError, match="Unknown top-level"):
            wc.load_config(cfg)

    def test_bad_entry_propagates(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("models:\n  - prajjwal1/bert-tiny\n")
        with pytest.raises(wc.ConfigError, match="must be pinned"):
            wc.load_config(cfg)
```

- [ ] **Step 2: Create the test package + sys.path shim**

```python
# tests/ci/__init__.py
```
(empty file)

```python
# tests/ci/conftest.py
"""Put the .ci/ directory on sys.path so tests can import warm_hf_cache.py."""
from __future__ import annotations

import sys
from pathlib import Path

_CI_DIR = Path(__file__).resolve().parents[2] / ".ci"
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/ci/ -v
```
Expected: collection error or `ModuleNotFoundError: No module named 'warm_hf_cache'`.

- [ ] **Step 4: Write the script skeleton with pure helpers**

```python
# .ci/warm_hf_cache.py
"""Pre-populate the HuggingFace cache for the autointent CI test suite.

Reads ``.ci/hf-prewarm.yaml`` and ensures every listed model / dataset is
present in ``~/.cache/huggingface`` at the pinned revision. The CI workflow
runs this in a dedicated job before any test job so tests find every HF
resource on disk and never make HF API calls themselves.

Entries are ``"<repo_id>@<sha>"`` where ``sha`` is a 40-char hex commit
hash. Pinning to a SHA lets the fast path (``snapshot_download`` with
``local_files_only=True``) decide "cache is complete" without calling the
HF API, which is what avoids the 1000-req/5-min rate limit on cold-cache
CI runs.

This module is also imported by ``tests/ci/test_warm_hf_cache.py``, so
keep top-level imports cheap and side-effect-free.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ConfigError(ValueError):
    """Raised when ``.ci/hf-prewarm.yaml`` is malformed."""


@dataclass(frozen=True)
class Entry:
    """One HF resource to prewarm."""

    repo_type: str  # "model" or "dataset"
    repo_id: str
    revision: str


def parse_entry(text: str) -> tuple[str, str]:
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


def load_config(path: Path) -> list[Entry]:
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
            repo, rev = parse_entry(raw)
            entries.append(Entry(repo_type=repo_type, repo_id=repo, revision=rev))
    return entries
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run --with pyyaml pytest tests/ci/ -v
```
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add .ci/warm_hf_cache.py tests/ci/__init__.py tests/ci/conftest.py tests/ci/test_warm_hf_cache.py
git commit -m "ci: add warm_hf_cache.py config loader + unit tests"
```

---

### Task 3: Warmer script — download logic + CLI

**Files:**
- Modify: `.ci/warm_hf_cache.py` (add `prewarm_entry`, `main`, CLI)

- [ ] **Step 1: Append the download logic and CLI to `.ci/warm_hf_cache.py`**

Append to the file (after the `load_config` function):

```python
import argparse
import logging
import sys
import time
from typing import Literal

logger = logging.getLogger("warm_hf_cache")

# Backoff sized for the HF rate limit window (per 5 min for authenticated
# users). Shorter waits usually hit the same throttle bucket and burn
# through retries; the totals here ride out two full windows in the worst
# case (60 + 120 + 180 = 360s).
_RETRY_DELAYS = (60, 120, 180)

Outcome = Literal["cached", "downloaded", "failed"]


def prewarm_entry(entry: Entry) -> Outcome:
    """Ensure ``entry`` is fully present in the local HF cache.

    Returns ``"cached"`` if every file was already on disk (no HF API
    contact at all), ``"downloaded"`` after a successful network pull, or
    ``"failed"`` if all retries were exhausted.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    label = f"{entry.repo_type}:{entry.repo_id}@{entry.revision[:8]}"

    # Fast path: every file already on disk → no API call.
    try:
        snapshot_download(
            repo_id=entry.repo_id,
            revision=entry.revision,
            repo_type=entry.repo_type,
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, FileNotFoundError, OSError):
        pass  # Fall through to network download.
    else:
        logger.info("%s — cached", label)
        return "cached"

    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            snapshot_download(
                repo_id=entry.repo_id,
                revision=entry.revision,
                repo_type=entry.repo_type,
            )
        except (HfHubHTTPError, OSError) as exc:
            logger.warning("%s — attempt %d failed (%s)", label, attempt, exc)
            if delay is None:
                logger.error("%s — giving up after %d attempts", label, attempt)
                return "failed"
            logger.info("%s — sleeping %ds before retry", label, delay)
            time.sleep(delay)
        else:
            logger.info("%s — downloaded", label)
            return "downloaded"
    return "failed"  # unreachable, keeps the type checker happy


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns 0 on success in non-strict mode (even when individual entries
    failed — best-effort by design). Returns non-zero only when ``--strict``
    is set and at least one entry failed, or when the config itself is
    malformed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".ci/hf-prewarm.yaml"),
        help="Path to the prewarm config YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any entry failed to prewarm.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        entries = load_config(args.config)
    except ConfigError as exc:
        logger.error("Config error: %s", exc)
        return 2

    counts: dict[Outcome, int] = {"cached": 0, "downloaded": 0, "failed": 0}
    for entry in entries:
        counts[prewarm_entry(entry)] += 1

    logger.info(
        "Summary: %d cached, %d downloaded, %d failed",
        counts["cached"],
        counts["downloaded"],
        counts["failed"],
    )
    if args.strict and counts["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint and format**

```bash
uv run ruff check .ci/warm_hf_cache.py
uv run ruff format .ci/warm_hf_cache.py --check
```
Expected: clean. If ruff flags issues, fix them. The `.ci/` path is NOT in the `tool.ruff.exclude` list so this file does get linted.

- [ ] **Step 3: Verify locally against the real config**

```bash
HF_TOKEN="${HF_TOKEN:-}" uv run --with 'huggingface_hub[hf_xet]' --with pyyaml python .ci/warm_hf_cache.py --strict
```
Expected: 12 entries report `cached` (since you've been running these tests locally) and the script exits 0. If anything reports `downloaded`, that's fine — it'll happen once then stay `cached`. If anything is `failed` even with `--strict`, investigate before continuing.

- [ ] **Step 4: Commit**

```bash
git add .ci/warm_hf_cache.py
git commit -m "ci: add warm_hf_cache.py download logic + CLI"
```

---

### Task 4: Restore-keys + drop the inline prewarm

**Files:**
- Modify: `.github/workflows/reusable-test.yaml`

- [ ] **Step 1: Replace the cache step's restore-keys cascade**

In `.github/workflows/reusable-test.yaml`, find this block:

```yaml
    - name: Cache Hugging Face
      id: cache-hf
      uses: actions/cache@v4
      with:
        path: ~/.cache/huggingface
        key: ${{ runner.os }}-hf-${{ github.run_id }}-${{ matrix.python-version }}
        restore-keys: |
          ${{ runner.os }}-hf-${{ github.run_id }}-
          ${{ runner.os }}-hf-
```

Replace the restore-keys with the prioritized cascade. The block becomes:

```yaml
    # The HF cache is shared across all reusable-test callers. The first
    # two restore-keys prefer caches produced by ci.yaml's warm-cache job
    # (the hashFiles key matches when `.ci/hf-prewarm.yaml` hasn't changed
    # between the warm-cache save and this restore). The fallback keys cover
    # branches whose warm-cache hasn't run yet.
    - name: Cache Hugging Face
      id: cache-hf
      uses: actions/cache@v4
      with:
        path: ~/.cache/huggingface
        key: ${{ runner.os }}-hf-${{ github.run_id }}-${{ matrix.python-version }}
        restore-keys: |
          ${{ runner.os }}-hf-warmed-${{ hashFiles('.ci/hf-prewarm.yaml') }}-
          ${{ runner.os }}-hf-warmed-
          ${{ runner.os }}-hf-${{ github.run_id }}-
          ${{ runner.os }}-hf-
```

- [ ] **Step 2: Delete the `prewarm_models` input declaration**

Remove this block from the `on.workflow_call.inputs` section:

```yaml
      prewarm_models:
        required: false
        type: string
        default: ''
        description: 'Newline-separated list of HF repos to pre-download with retries before tests run. Each entry: "[type:]repo_id[@revision]" where type is "model" (default) or "dataset", and revision defaults to main.'
```

- [ ] **Step 3: Delete the inline prewarm step**

Remove the entire `- name: Pre-warm Hugging Face models (best-effort)` step (about 60 lines of bash). The step that runs `uv sync` is followed directly by the `- name: Run tests` step.

- [ ] **Step 4: Verify the file still parses as YAML**

```bash
uv run --with pyyaml python -c "import yaml; yaml.safe_load(open('.github/workflows/reusable-test.yaml'))"
```
Expected: no output (success).

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/reusable-test.yaml
git commit -m "ci: prefer warmed cache in reusable-test; drop inline prewarm"
```

---

### Task 5: Orchestrator workflow

**Files:**
- Create: `.github/workflows/ci.yaml`

- [ ] **Step 1: Write `.github/workflows/ci.yaml`**

```yaml
name: CI

on:
  push:
    branches:
      - dev
  pull_request:

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  warm-cache:
    name: Warm HF cache
    strategy:
      fail-fast: false
      matrix:
        os: [ ubuntu-latest, windows-latest ]
    runs-on: ${{ matrix.os }}
    # Best-effort: an HF outage here must not cancel every PR. Test jobs
    # `needs: warm-cache` but proceed regardless via continue-on-error.
    continue-on-error: true
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Cache Hugging Face
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          # Cache key is bound to the hash of `.ci/hf-prewarm.yaml` so an
          # edit to the list invalidates exactly one level of the cascade;
          # the next-most-specific restore-key still catches the previous
          # warmed cache so only the diff is re-downloaded.
          key: ${{ runner.os }}-hf-warmed-${{ hashFiles('.ci/hf-prewarm.yaml') }}-${{ github.run_id }}
          restore-keys: |
            ${{ runner.os }}-hf-warmed-${{ hashFiles('.ci/hf-prewarm.yaml') }}-
            ${{ runner.os }}-hf-warmed-
            ${{ runner.os }}-hf-

      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          version: "0.10.0"

      - name: Run warm-cache script
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: uv run --with 'huggingface_hub[hf_xet]' --with pyyaml python .ci/warm_hf_cache.py

  unit-tests:
    name: unit-tests
    needs: warm-cache
    uses: ./.github/workflows/reusable-test.yaml
    secrets: inherit
    with:
      test_command: pytest -n auto --ignore=tests/modules/scoring/ --ignore=tests/pipeline --ignore=tests/embedder

  test-embedder:
    name: test-embedder
    needs: warm-cache
    uses: ./.github/workflows/reusable-test.yaml
    secrets: inherit
    with:
      test_command: pytest -n auto tests/embedder/ tests/callback/
      extras: --extra sentence-transformers --extra transformers

  test-scorers:
    name: test-scorers
    needs: warm-cache
    strategy:
      fail-fast: false
      matrix:
        dependency-group: [ "base", "transformers", "peft", "catboost" ]
    uses: ./.github/workflows/reusable-test.yaml
    secrets: inherit
    with:
      test_command: pytest -n auto tests/modules/scoring/
      extras: ${{ matrix.dependency-group != 'base' && format('--extra {0}', matrix.dependency-group) || '' }}

  test-presets:
    name: test-presets
    needs: warm-cache
    uses: ./.github/workflows/reusable-test.yaml
    secrets: inherit
    with:
      test_command: pytest -n auto tests/pipeline/test_presets.py
      extras: --extra catboost --extra peft --extra transformers --extra sentence-transformers

  test-optimization:
    name: test-optimization
    needs: warm-cache
    uses: ./.github/workflows/reusable-test.yaml
    secrets: inherit
    with:
      test_command: pytest -n auto tests/pipeline/test_optimization.py
      extras: --extra catboost --extra peft --extra transformers --extra sentence-transformers

  test-inference:
    name: test-inference
    needs: warm-cache
    uses: ./.github/workflows/reusable-test.yaml
    secrets: inherit
    with:
      test_command: pytest -n auto tests/pipeline/test_inference.py
      extras: --extra catboost --extra peft --extra transformers --extra sentence-transformers
```

Note one schema detail: GitHub Actions does not allow `matrix` on reusable-workflow callers in some older docs, but as of 2024+ this is supported. The `test-scorers` job above uses it.

- [ ] **Step 2: Verify the file still parses as YAML**

```bash
uv run --with pyyaml python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yaml'))"
```
Expected: no output (success).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yaml
git commit -m "ci: add orchestrator workflow with warm-cache gating test jobs"
```

---

### Task 6: Delete the old test-*.yaml files

**Files:**
- Delete: `.github/workflows/unit-tests.yaml`
- Delete: `.github/workflows/test-embedder.yaml`
- Delete: `.github/workflows/test-scorers.yaml`
- Delete: `.github/workflows/test-presets.yaml`
- Delete: `.github/workflows/test-optimization.yaml`
- Delete: `.github/workflows/test-inference.yaml`

- [ ] **Step 1: Delete the six files**

```bash
git rm .github/workflows/unit-tests.yaml \
       .github/workflows/test-embedder.yaml \
       .github/workflows/test-scorers.yaml \
       .github/workflows/test-presets.yaml \
       .github/workflows/test-optimization.yaml \
       .github/workflows/test-inference.yaml
```

- [ ] **Step 2: Confirm remaining workflow files**

```bash
ls .github/workflows/
```
Expected: `build-docs.yaml`, `check-schema.yaml`, `ci.yaml`, `release.yaml`, `reusable-test.yaml`, `ruff.yml`, `typing.yml`.

- [ ] **Step 3: Commit**

```bash
git commit -m "ci: drop standalone test-*.yaml files (folded into ci.yaml)"
```

---

### Task 7: Push and observe

- [ ] **Step 1: Push the branch**

```bash
git push
```

- [ ] **Step 2: Watch the first warm-cache run**

```bash
gh run watch $(gh run list --branch b/hf-rate-limit-on-tests --workflow CI --limit 1 --json databaseId --jq '.[0].databaseId')
```
Expected on the first run:
- `warm-cache (ubuntu-latest)` and `warm-cache (windows-latest)` both start.
- Each restores some `Linux-hf-` / `Windows-hf-` cache via the wildcard fallback (whatever's left over from the per-branch caches built by earlier failed runs).
- The script logs a mix of `cached` and `downloaded` lines, with a summary like `X cached, Y downloaded, 0 failed`.
- Total wall time: 5-10 minutes on Linux, similar on Windows.
- All six test jobs start once warm-cache completes and pass cleanly with cache hits.

- [ ] **Step 3: Trigger a second run to verify steady state**

```bash
git commit --allow-empty -m "ci: trigger steady-state cache verification"
git push
```

Expected on the second run:
- `warm-cache` restores the exact-hash key from the previous run → cache hit.
- The script logs all entries as `cached`.
- Summary: `12 cached, 0 downloaded, 0 failed`.
- warm-cache step duration: ~30 seconds (mostly uv setup).
- All test jobs cache-hit and run.

- [ ] **Step 4: Update branch protection (out-of-band)**

PR status check names changed from e.g. `unit tests / test (ubuntu-latest, 3.10)` to `CI / unit-tests / test (ubuntu-latest, 3.10)`. Repo admin must update the protected-branch rules on `dev` to reference the new check names. **This is a manual step**; mention it in the PR description so the reviewer doesn't merge until the rules are updated. If branch protection blocks merge, the PR can either wait for rule updates or admin-merge.

---

## Self-Review Notes

- **Spec coverage:** every section of `docs/superpowers/specs/2026-06-05-hf-cache-warmer-design.md` maps to a task (architecture → Tasks 1-6; components → Tasks 1-5; data flow → Task 7 verification; error handling → Task 3's logic; testing → Tasks 2 and 7).
- **Placeholders:** none. The `<lookup>` placeholders from the spec are resolved with real SHAs in Task 1.
- **Type consistency:** `Entry`, `parse_entry`, `load_config`, `prewarm_entry`, `Outcome`, `ConfigError` are defined consistently across Tasks 2 and 3. The CLI in Task 3 uses `load_config` and `prewarm_entry` exactly as defined.
- **Scope:** focused — one orchestrator workflow, one Python script, one config file. Branch protection rule update is correctly flagged as out-of-band.
