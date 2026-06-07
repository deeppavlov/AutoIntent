# HF SHA Single Source of Truth — Design Spec

**Status:** Approved (Option A from brainstorm on 2026-06-07)

**Goal:** Make `DEFAULT_REVISIONS` the only place a HuggingFace commit SHA is written. Every other surface that needs a pinned SHA derives it from there.

**Architecture:** Extract the pin dict to a tiny zero-dependency leaf module. Auto-fill consumers (HFModelConfig validator) keep working. CI prewarm YAMLs lose their SHA suffixes and carry repo IDs only; the warm script joins repo IDs to SHAs at runtime. Test files and search-space YAMLs drop their redundant `revision=` lines and let the validator fill them. A single test asserts the prewarm subset is a subset of `DEFAULT_REVISIONS`.

---

## 1. Background

`DEFAULT_REVISIONS` already exists as the *intended* single source of truth in `src/autointent/configs/_transformers.py`:

```python
DEFAULT_REVISIONS: dict[str, str] = {
    "prajjwal1/bert-tiny": "79779625a0a40f1eee8496e16056bc0d7766df22",
    ...  # 11 entries total
}
```

`HFModelConfig._apply_default_revision` is a `model_validator(mode="after")` that fills `revision` from this dict whenever a user instantiates a config without an explicit revision. So in principle, every consumer downstream of `HFModelConfig` is already covered.

The problem: four other surfaces hard-code the same SHA strings literally, none of which need to:

| Surface | Today | Why it's redundant |
|---|---|---|
| `.ci/hf-prewarm-{linux,windows}.yaml` | `prajjwal1/bert-tiny@79779625...` × 3 entries × 2 files | The SHA is already in `DEFAULT_REVISIONS` |
| `tests/modules/scoring/test_{bert,lora,ptuning}.py` | `HFModelConfig(model_name="...", revision="79779625...")` | Validator auto-fills if `revision=None` |
| `tests/assets/configs/{multiclass,multilabel}.yaml` | `revision: 79779625...` (4 entries) | Same — the YAML is loaded into `HFModelConfig` |
| `tests/ci/test_warm_hf_cache.py` | SHAs in parser fixture strings | Parser semantics change (see §6.4) |

Total: ~13 duplicate SHA strings to remove.

---

## 2. Goals & Non-Goals

**Goals:**

1. After this refactor, the *only* file you edit to bump a pinned SHA is `DEFAULT_REVISIONS`.
2. Adding a new pinned model is two lines: one in `DEFAULT_REVISIONS`, optionally one in `.ci/hf-prewarm-linux.yaml` if it should be CI-prewarmed.
3. CI catches drift between the prewarm subset and `DEFAULT_REVISIONS` automatically.
4. The warm-cache job's import cost stays the same (no `uv sync` of autointent's full dep tree).

**Non-goals:**

- Moving `DEFAULT_REVISIONS` to a data file (TOML/JSON). Deferred to a future spec if a non-Python consumer ever appears.
- Adding bot-driven SHA bumps (Renovate/Dependabot). Out of scope.
- Changing the public API of `HFModelConfig` / `EmbedderConfig` / `CrossEncoderConfig`. The validator behavior is preserved verbatim.
- Restructuring `src/autointent/_presets/*.yaml`. Those are user-facing; they don't carry SHAs and aren't touched.

---

## 3. Architecture

### 3.1 Leaf module for the dict

Extract `DEFAULT_REVISIONS` (only the dict, plus its docstring) from `src/autointent/configs/_transformers.py` to a new zero-dep leaf module:

```
src/autointent/configs/_pinned_revisions.py
```

This module has **no imports** other than `from __future__ import annotations`. It is loadable from a script that has not installed autointent, via a one-line `sys.path` shim (see §3.3).

`_transformers.py` re-exports the symbol so all existing consumers keep working without changes:

```python
# src/autointent/configs/_transformers.py
from autointent.configs._pinned_revisions import DEFAULT_REVISIONS  # noqa: F401
```

### 3.2 CI prewarm YAMLs lose their SHAs

`.ci/hf-prewarm-linux.yaml` (and `-windows.yaml`, which is identical content today):

```yaml
# Repo IDs only. The pinned SHAs live in
# src/autointent/configs/_pinned_revisions.py::DEFAULT_REVISIONS and are
# joined in by warm_hf_cache.py at runtime. Adding a repo here without
# adding a matching DEFAULT_REVISIONS entry fails the cross-check test.
models:
  - prajjwal1/bert-tiny
  - cross-encoder/ms-marco-MiniLM-L6-v2
  - sergeyzh/rubert-tiny-turbo
datasets: []
```

### 3.3 `warm_hf_cache.py` joins repo IDs to SHAs

`.ci/warm_hf_cache.py` gets a tiny path shim to import the leaf module, and changes the parser to look up the SHA instead of parsing one:

```python
# .ci/warm_hf_cache.py - near the top, after stdlib imports
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from autointent.configs._pinned_revisions import DEFAULT_REVISIONS  # noqa: E402
```

The parser changes from `_parse_entry(text: str) -> (repo, sha)` to:

```python
def _resolve_entry(repo_id: str, repo_type: str) -> Entry:
    """Resolve a repo ID to a pinned Entry via DEFAULT_REVISIONS.

    Raises ConfigError if repo_id is not pinned. Datasets are not in
    DEFAULT_REVISIONS today (the dict is models-only); if the warm config
    ever lists a dataset, this function must be extended.
    """
    if repo_id not in DEFAULT_REVISIONS:
        msg = (
            f"{repo_id!r} not in DEFAULT_REVISIONS. Add an entry to "
            "src/autointent/configs/_pinned_revisions.py before listing it here."
        )
        raise ConfigError(msg)
    return Entry(repo_type=repo_type, repo_id=repo_id, revision=DEFAULT_REVISIONS[repo_id])
```

`_load_config` calls `_resolve_entry` instead of `_parse_entry` per item.

**Dataset handling.** `DEFAULT_REVISIONS` covers models only today. The prewarm YAMLs' `datasets:` list is empty after the test refactor that just landed. If we ever need to warm a dataset, we extend either `DEFAULT_REVISIONS` (preferred — it stays the single source) or add a parallel `DEFAULT_DATASET_REVISIONS`. The spec defers this until it actually matters; the parser raises `ConfigError` if a dataset appears, so we can't silently regress.

### 3.4 Test/YAML cleanup

These files drop redundant SHA strings; the validator fills them.

**Python tests** (3 files, 1 line each):

```python
# Before
_config = HFModelConfig(model_name="prajjwal1/bert-tiny", revision="79779625a0a40f1eee8496e16056bc0d7766df22")
# After
_config = HFModelConfig(model_name="prajjwal1/bert-tiny")
```

**YAML search spaces** (`tests/assets/configs/multiclass.yaml`, `multilabel.yaml`, 4 entries total):

```yaml
# Before
classification_model_config:
  - model_name: prajjwal1/bert-tiny
    revision: 79779625a0a40f1eee8496e16056bc0d7766df22
# After
classification_model_config:
  - model_name: prajjwal1/bert-tiny
```

The conftest `_rewrite_field` helper already drops `revision` when retargeting `model_name` (added during the prior test refactor), so removing these explicit pins doesn't regress any downstream behavior.

### 3.5 Cross-check test

A new test under `tests/ci/` asserts the warm-cache YAMLs reference only repo IDs that are pinned in `DEFAULT_REVISIONS`:

```python
def test_prewarm_yamls_are_subset_of_default_revisions():
    """Every model in .ci/hf-prewarm-*.yaml must be in DEFAULT_REVISIONS."""
    import yaml
    from autointent.configs._pinned_revisions import DEFAULT_REVISIONS

    for path in (".ci/hf-prewarm-linux.yaml", ".ci/hf-prewarm-windows.yaml"):
        data = yaml.safe_load(Path(path).read_text())
        missing = [m for m in (data.get("models") or []) if m not in DEFAULT_REVISIONS]
        assert not missing, f"{path}: {missing} not in DEFAULT_REVISIONS"
```

This guarantees you can't add a repo to the prewarm config without also pinning its SHA.

### 3.6 `test_warm_hf_cache.py` updates

The parser tests change shape because the parser changes shape:

| Old test | New test |
|---|---|
| `test_valid_sha` — parses `repo@sha` | DELETE (no longer the parser's job) |
| `test_missing_sha_raises` — rejects bare repo ID | DELETE (bare repo ID is now the **valid** form) |
| `test_non_hex_revision_raises` | DELETE |
| `test_short_revision_raises` | DELETE |
| — | `test_known_repo_resolves` — `_resolve_entry("prajjwal1/bert-tiny", "model")` returns `Entry(..., revision=DEFAULT_REVISIONS["prajjwal1/bert-tiny"])` |
| — | `test_unknown_repo_raises` — unknown repo ID raises `ConfigError` with "not in DEFAULT_REVISIONS" |
| `test_empty_file` etc. | KEEP — YAML loader contract is unchanged |
| `test_load_config_*` with `repo@sha` fixtures | UPDATE — fixtures use bare repo IDs |

---

## 4. Migration safety

This refactor touches surfaces that are independently testable:

1. **Leaf-module extraction** is a pure cut/paste with a re-export shim. All existing imports of `DEFAULT_REVISIONS` keep working. Verify by running the test suite once before any other change.
2. **YAML format change** is breaking for `warm_hf_cache.py`, so the YAML and the script change in the same commit. The CI warm-cache job is the only consumer.
3. **Test/YAML SHA removal** is a no-op behaviorally because the validator was already filling `revision` whenever absent. Verify by running affected tests both before and after — they must pass identically.
4. **Cross-check test** is a new safety net; it can land at any point.

No production code path changes. The validator behavior is preserved verbatim. The `HFModelConfig.revision` field semantics are unchanged.

---

## 5. Risk & rollback

**Risk: `sys.path` shim in `warm_hf_cache.py` regresses.**
Mitigation: the shim is two lines; the test `tests/ci/test_warm_hf_cache.py` imports `warm_hf_cache as wc` already, so any import failure surfaces immediately under `pytest tests/ci/`. CI runs this suite on every PR.

**Risk: an existing consumer of `DEFAULT_REVISIONS` imports from a path we missed.**
Mitigation: `grep -rn DEFAULT_REVISIONS src/ tests/` returns 4 files today (verified). The re-export in `_transformers.py` covers the `autointent.configs._transformers.DEFAULT_REVISIONS` import path used by `tests/configs/test_combined_config.py` and `tests/conftest.py`.

**Risk: someone adds a new pin to `DEFAULT_REVISIONS` and the new leaf module gains an import.**
Mitigation: a ruff/lint rule in the leaf module's docstring saying "this module has no imports other than `from __future__`". Optionally, a test that asserts the module's `__file__` has no non-stdlib imports (overkill — the docstring banner plus reviewer discipline is enough).

**Rollback:** revert the single commit per phase (§6). Each phase is a clean checkpoint.

---

## 6. Migration order

Each phase is independently green-CI:

1. **Phase 1 — extract leaf module.** Create `src/autointent/configs/_pinned_revisions.py` with the dict; replace the dict in `_transformers.py` with `from autointent.configs._pinned_revisions import DEFAULT_REVISIONS`. No other changes. Full test suite passes unchanged.
2. **Phase 2 — flip warm script + prewarm YAMLs.** Rewrite `_parse_entry` → `_resolve_entry`; rewrite both `.ci/hf-prewarm-*.yaml` to repo-ID-only; update `tests/ci/test_warm_hf_cache.py` per §3.6. Run `pytest tests/ci/` locally to verify; run the warm-cache job on CI to verify against real HF cache.
3. **Phase 3 — drop redundant SHAs from tests/YAMLs.** Edit the 3 Python test files and 2 YAML files per §3.4. Run the affected test files to verify behavior is unchanged.
4. **Phase 4 — add cross-check test.** Add the test from §3.5. Verify it catches the drift case by temporarily adding a fake repo to a prewarm YAML.

Total: ~4 commits, each independently green and revertable.

---

## 7. Out-of-scope follow-ups

These come up during this work but should NOT be part of this refactor:

- The two `.ci/hf-prewarm-{linux,windows}.yaml` files are identical content today. Collapsing them to one file with the CI matrix passing the same path is a separate cleanup; keep the per-OS split here so the diff stays focused.
- Doc-mode SHA references in `docs/superpowers/specs/` and `docs/superpowers/plans/` are historical snapshots of plan state — they should not be rewritten.
- `tests/ci/test_warm_hf_cache.py` fixture SHAs in the YAML-loader tests (not the parser tests) become incidental — they're testing that "models: [<bare repo id>]" deserializes correctly, not that the SHA shape is enforced. Those tests get simpler too but the change is mechanical, not architectural.

---

## 8. Acceptance criteria

After all four phases land:

- `grep -rn -E '[0-9a-f]{40}' src/ tests/ .ci/ --include='*.py' --include='*.yaml'` returns exactly:
  - `src/autointent/configs/_pinned_revisions.py` (the dict — the single source)
  - `tests/configs/test_combined_config.py` (assertions that *read* the dict — fine)
  - `tests/ci/test_warm_hf_cache.py` (fixtures that *read* DEFAULT_REVISIONS — fine)
  - any other test that explicitly *checks* a SHA is correct — fine
- No `.yaml` file under `.ci/` or `tests/assets/configs/` contains a literal 40-hex SHA.
- The test suite passes on `pytest -n auto`.
- The CI warm-cache job completes successfully with the new repo-ID-only YAML.
- Adding a fake unpinned repo to a prewarm YAML fails the cross-check test from §3.5.
