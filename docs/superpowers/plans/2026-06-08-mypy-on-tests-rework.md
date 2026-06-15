# Phase D — kwargs upstream + cast→assert/TypeGuard rework (mypy-on-tests)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ~23 of the 34 test-side `# type: ignore`s opened by Phase B by fixing 5 `src/` `**kwargs: dict[str, Any]` annotation bugs upstream (CatBoost `__init__` + `from_context`, Sklearn `__init__` + `from_context`, Generator `__init__`), and replace test-side `cast(...)` calls with `assert isinstance(...)` narrowing or shared `tests/_helpers/typeguards.py` predicates where natural. (The 34→~23 accounting is over *test-side* ignores added by Phase B; two pre-existing src-side ignores in `_generator.py:235,322` were never part of that pool and stay in place.)

**Architecture:** Two-phase rework sitting between Phase B (subagent fan-out, 10 open PRs against `b/mypy-on-tests`) and Phase C (final integration + gate flip). Phase D1 lands src/ + helper changes on `b/mypy-on-tests` in one commit (main thread). Phase D2 fans out 10 parallel subagents — each on its own existing PR branch — that rebase, drop now-unused ignores, and convert `cast(...)`s to `assert isinstance(...)` / `typeguards.*` predicates as appropriate. Phase D3 serial-merges the refreshed PRs as CI goes green. The originating `b/mypy-on-tests` Phase C plan (`docs/superpowers/plans/2026-06-07-mypy-on-tests.md`) resumes at Phase C unchanged after Phase D completes.

**Tech Stack:** Python 3.10+ (uses `typing.TypeGuard`, not `typing.TypeIs` which is 3.13+), mypy 1.x (strict), pydantic v2 mypy plugin, pytest 8.x, uv 0.10 for environment + worktree management, git worktrees + GitHub PRs for subagent isolation.

**Parent plan:** `docs/superpowers/plans/2026-06-07-mypy-on-tests.md` — read its "Subagent contract template" and "Parallelism note" before touching this. The hard-limit list in that contract still applies here, with one deliberate amendment in D1 (this commit DOES add `tests/_helpers/typeguards.py` — that addition is the whole point of D1; the existing files in `_helpers` remain locked).

**Verification rules (carry forward from parent plan):**
- DO NOT run pytest locally — not even `--collect-only`. CI on each PR is the verification surface. (The original OOM that caused this entire constraint is documented in the parent spec.)
- Mypy + ruff are fine to run locally.

---

## File Structure

| Path | Touched by | Responsibility |
|---|---|---|
| `src/autointent/modules/scoring/_catboost/catboost_scorer.py` | D1 | Fix `**catboost_kwargs: dict[str, Any]` → `**catboost_kwargs: Any` on `__init__` (line 110) and `from_context` (line 143). |
| `src/autointent/modules/scoring/_sklearn/sklearn_scorer.py` | D1 | Fix `**clf_args: dict[str, float \| str \| bool]` → `**clf_args: Any` on `__init__` (line 70) **and** `from_context` (line 98). The review caught that the original plan missed the second signature. |
| `src/autointent/generation/_generator.py` | D1 | Fix `**generation_params: dict[str, Any]` → `**generation_params: Any` on `__init__` (line 131). |
| _Out of scope for D1:_ `src/autointent/_callbacks/{base,emissions_tracker,callback_handler,wandb,tensorboard}.py` `log_value` methods and `src/autointent/modules/base/_base.py:48,118,130,145` have the same `**kwargs: dict[str, Any]` pattern. NOT fixed in D1 because the Phase B subagents did NOT add any `[arg-type]` ignores for `log_value` or BaseModule kwargs — so widening these would not eliminate any current ignore. Deferred to a separate workstream if/when CI surfaces them. | — | — |
| `tests/_helpers/typeguards.py` | D1 (created) | New: `is_strict_labels(labels: ListOfGenericLabels) -> TypeGuard[ListOfLabels]` (and any other narrow-from-union TypeGuards subagents flagged as recurring). |
| `tests/_helpers/__init__.py` | D1 | Re-export `is_strict_labels` so subagents import `from tests._helpers import is_strict_labels`. |
| `tests/modules/scoring/**/*.py` (PR #313) | D2 / B1 | Subagent B1: drop `[arg-type]` ignores invalidated by catboost/sklearn fix; replace remaining casts with asserts/TypeGuard. |
| `tests/modules/decision/**/*.py` (PR #304) | D2 / B2 | Subagent B2: replace `cast("ListOfLabels", ...)` with `is_strict_labels` TypeGuard. |
| `tests/modules/{embedding,test_dumper.py,test_regex.py}` (PR #307) | D2 / B3 | Subagent B3: replace subclass casts with `assert isinstance(...)`. |
| `tests/embedder/**/*.py` (PR #312) | D2 / B4 | Subagent B4: replace `cast("...Config", ...)` with `assert isinstance(...)`; replace `cast("ListOfLabels", ...)` with `is_strict_labels`. |
| `tests/data/**/*.py` (PR #306) | D2 / B5 | Subagent B5: review; likely no-op (no casts were used). |
| `tests/generation/**/*.py` (PR #311) | D2 / B6 | Subagent B6: drop `[arg-type]` ignores invalidated by generator fix. |
| `tests/configs/**/*.py` (PR #308) | D2 / B7 | Subagent B7: review; keep `cast("TaskType", "full_training")` (still requires Phase C decision on widening the Literal). |
| `tests/pipeline/**/*.py` (PR #310) | D2 / B8 | Subagent B8: replace `cast("NodeOptimizer", node)` and `cast("...EmbeddingConfig", ...)` with asserts. KEEP `cast("Any", ...)` for deliberate private-state poke. |
| `tests/context/**/*.py` (PR #305) | D2 / B9 | Subagent B9: replace `cast("OpenSearchBackend", ...)` and `cast("HashingVectorizerEmbeddingConfig", ...)` with asserts (the surrounding `isinstance` guards already exist — just add a redundant narrowing assert or drop the cast). |
| `tests/{callback,ci,metrics}/**/*.py` (PR #309) | D2 / B10 | Subagent B10: 2 `[arg-type]` ignores stay (they're for the separate `@ignore_oos` decorator gap — Phase C decision, NOT in D1's scope). |

Files NOT touched by this plan: anything else in `src/` (the `@ignore_oos` decorator gap and other type-widenings remain Phase C decisions); the docs; the workflows.

---

## Conventions

- **All commands in D1 run from**: `/Users/voorhs/repos/lab/AutoIntent/.claude/worktrees/mypy-on-tests`.
- **All commands in D2 run from**: each subagent's own worktree (Agent re-uses or recreates one).
- **Worktree quirk reminder**: in Phase B several subagents reported that `Agent({isolation: "worktree"})` forked from `dev` rather than `b/mypy-on-tests`. They handled it correctly by resetting to `origin/b/mypy-on-tests` first. The D2 contract includes an explicit reset/rebase step so the next round can't trip on this.
- **Sync command**: `uv sync --group typing --group test --extra catboost --extra peft --extra transformers --extra sentence-transformers --extra openai`. This is the same `--group typing --group test` invocation that CI now uses (after Phase A's commit `1bd1d989`).
- **Parent spec wins**: when this plan and the parent spec (`docs/superpowers/specs/2026-06-07-mypy-on-tests-design.md`) disagree, the spec wins; flag the inconsistency.

---

# Phase D1 — Land kwargs fix + typeguards.py on `b/mypy-on-tests`

Sequential, main thread, one commit. After this commit pushes, the 10 open PRs each need to rebase (D2).

## Task D1.1: Fix `**kwargs: dict[str, Any]` → `**kwargs: Any` in 3 src/ files

**Files:**
- Modify: `src/autointent/modules/scoring/_catboost/catboost_scorer.py` — two signatures at lines 110 and 143.
- Modify: `src/autointent/modules/scoring/_sklearn/sklearn_scorer.py` — **two** signatures at lines 70 and 98.
- Modify: `src/autointent/generation/_generator.py` — one signature at line 131.

Confirm line numbers before editing by running:
```bash
grep -nE '\*\*[a-z_]+: dict' src/autointent/modules/scoring/_catboost/catboost_scorer.py src/autointent/modules/scoring/_sklearn/sklearn_scorer.py src/autointent/generation/_generator.py
```
Expected output (5 lines, matching the line numbers above).

- [ ] **Step 1: Edit `_catboost/catboost_scorer.py` — `__init__` (line 110)**

```python
# before
**catboost_kwargs: dict[str, Any],
# after
**catboost_kwargs: Any,
```

- [ ] **Step 2: Edit `_catboost/catboost_scorer.py` — `from_context` (line 143)**

Same replacement as Step 1.

- [ ] **Step 3a: Edit `_sklearn/sklearn_scorer.py` — `__init__` (line 70)**

```python
# before
**clf_args: dict[str, float | str | bool],
# after
**clf_args: Any,
```

- [ ] **Step 3b: Edit `_sklearn/sklearn_scorer.py` — `from_context` (line 98)**

Same replacement as Step 3a. (Reviewer caught this — the original plan missed this signature and would have left a class of `[arg-type]` ignores in PR #313 that the rebase couldn't drop.)

- [ ] **Step 4: Edit `_generator.py` — `Generator.__init__` (line 131)**

```python
# before
**generation_params: dict[str, Any],
# after
**generation_params: Any,
```

- [ ] **Step 5: Verify mypy on src/ is still clean**

```bash
uv run --group typing --group test mypy src/autointent 2>&1 | tail -3
```
Expected: `Success: no issues found in N source files`.

If this fails, inspect the error. The change widens kwarg typing (more permissive), so any new errors are likely in code that was *relying* on the bug — investigate before patching downstream.

---

## Task D1.2: Create `tests/_helpers/typeguards.py`

**Files:**
- Create: `tests/_helpers/typeguards.py`
- Modify: `tests/_helpers/__init__.py`

- [ ] **Step 1: Create `tests/_helpers/typeguards.py`**

```python
"""TypeGuard predicates used across test code to narrow union types from src/.

Each predicate has a runtime check that is cheap and a TypeGuard return type that
mypy uses to narrow the calling scope. Prefer these over `typing.cast` when the
narrowing has a verifiable runtime invariant the test relies on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    from autointent.custom_types import ListOfGenericLabels, ListOfLabels


def is_strict_labels(labels: ListOfGenericLabels) -> TypeGuard[ListOfLabels]:
    """True iff every label is non-None (i.e. no OOS samples).

    Narrows `ListOfGenericLabels` (= `ListOfLabels | ListOfLabelsWithOOS`) to
    `ListOfLabels` so the call site can pass it to APIs that require the
    non-OOS variant (e.g. `Embedder.train`, `SentenceTransformerEmbeddingBackend.train`,
    `KNNScorer.fit`).
    """
    return all(label is not None for label in labels)
```

- [ ] **Step 2: Write `tests/_helpers/__init__.py`**

The current file is empty (0 bytes — verified). Write it fresh:

```python
"""Test helpers."""

from tests._helpers.typeguards import is_strict_labels

__all__ = ["is_strict_labels"]
```

No need to preserve anything; nothing is there. Future helpers append to `__all__`.

- [ ] **Step 3: Verify mypy still clean on `tests/_helpers`**

```bash
uv run --group typing --group test mypy tests/_helpers 2>&1 | tail -3
```
Expected: `Success: no issues found in N source files` (N grew by 1 from the new file).

If mypy fails on the pre-existing `tests/_helpers/extract_clinc150_subset.py` (a one-shot helper that wasn't actively type-checked before this rework), and the failure is unrelated to your new code, that is pre-existing drift — STOP, do not paper over it with new ignores. Either fix it as a minimal commit immediately preceding D1.4 or revert D1 entirely and escalate. Failures on `typeguards.py` or `__init__.py` are your code; fix them.

- [ ] **Step 4: Verify ruff clean on the new file**

```bash
uv tool run ruff check tests/_helpers
uv tool run ruff format --check tests/_helpers
```
Expected: `All checks passed!` for both.

If ruff suggests fixes, apply them and re-verify.

---

## Task D1.3: Verify the full Phase A surface still passes

**Files:** none modified.

- [ ] **Step 1: Run mypy on the integration surface**

```bash
uv run --group typing --group test mypy src/autointent tests/conftest.py tests/_fixtures tests/_helpers tests/_transformers 2>&1 | tail -3
```
Expected: `Success: no issues found in N source files`.

- [ ] **Step 2: Run ruff on the integration surface**

```bash
uv tool run ruff check src/autointent tests/conftest.py tests/_fixtures tests/_helpers tests/_transformers
```
Expected: `All checks passed!`.

If either fails, stop and fix before committing.

---

## Task D1.4: Commit and push D1

**Files:** all D1.1 + D1.2 changes.

- [ ] **Step 1: Stage all changes**

```bash
git add src/autointent/modules/scoring/_catboost/catboost_scorer.py \
        src/autointent/modules/scoring/_sklearn/sklearn_scorer.py \
        src/autointent/generation/_generator.py \
        tests/_helpers/typeguards.py \
        tests/_helpers/__init__.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
fix(src+tests): widen **kwargs:dict[str,Any]→Any in 3 scorers/generator; add tests/_helpers/typeguards.is_strict_labels

Phase B subagents discovered three identical src/ signature bugs where
**kwargs was typed as `dict[str, Any]` (one dict per kwarg value) instead
of `Any` (per-value). The annotation rejected scalar kwargs like
`max_tokens=1000` or `learning_rate=0.05` even though they're valid at
runtime. Affects CatBoostScorer.__init__ + .from_context, SklearnScorer.__init__,
and Generator.__init__.

Also adds tests/_helpers/typeguards.is_strict_labels for the recurring
ListOfGenericLabels -> ListOfLabels narrowing pattern, replacing one
class of test-side typing.cast usages with a verified runtime predicate.

Open Phase B PRs (#304-#313) rebase onto this commit to drop ~25 of the
34 # type: ignore additions; rework happens in Phase D2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Push**

```bash
git push origin b/mypy-on-tests
```

This refreshes the base branch on GitHub. Each open Phase B PR will now show "N commits behind" on the GitHub UI; D2 fixes that per-PR.

---

# Phase D2 — Per-PR rework, 10 parallel subagents

Each subagent owns one of the 10 open PRs from Phase B. The work shape is identical across PRs: rebase, drop now-unused ignores, replace casts with asserts or TypeGuards. Subagents fan out as a single message of 10 parallel `Agent` calls, mirroring the Phase B fan-out.

## Subagent contract template (D2)

When dispatching a D2 subagent, the main thread sends:

```
You are a Phase D2 subagent in the strict-mypy-on-tests workstream. Your worktree was just forked from `b/mypy-on-tests` (Phase D1 just landed src/ kwargs fixes and tests/_helpers/typeguards.py).

YOUR ASSIGNMENT: refresh PR #<PR-N> on branch `<existing-subagent-branch>`. The PR is from Phase B and needs two things:
1. Rebase onto the updated `b/mypy-on-tests` so the new src/ kwargs fix + typeguards.py are present.
2. Drop `# type: ignore`s that are now unused (mypy's `warn_unused_ignores` will flag them after the rebase).
3. Replace `typing.cast(T, x)` usages with `assert isinstance(x, T)` narrowing or `tests._helpers.typeguards.*` predicates where natural — see rules below.

================================================================
CRITICAL CONSTRAINT — DO NOT RUN PYTEST LOCALLY, NOT EVEN `--collect-only`
================================================================
The original Phase B execution OOM'd the host when 10 parallel subagents simultaneously imported torch / transformers / sentence-transformers via `pytest --collect-only`. Verification of pytest behavior happens on GitHub Actions via your refreshed PR — NOT locally. Mypy alone is fine to run locally.

REQUIRED READING (in your worktree, in order):
1. `docs/superpowers/plans/2026-06-08-mypy-on-tests-rework.md` — this plan, especially:
   - "Subagent contract template (D2)" (you're inside it now)
   - "Cast → assert/TypeGuard rules" (below)
2. `docs/superpowers/specs/2026-06-07-mypy-on-tests-design.md` — parent spec. The 7-item policy still applies to any *new* typing decisions.

CAST → ASSERT/TYPEGUARD RULES:
- **Pattern A: subclass-from-base narrowing where a runtime check exists nearby.**
  Replace `x = cast("Sub", base_obj)` with `assert isinstance(base_obj, Sub)` and use `base_obj` directly (mypy narrows).
  Example: `cast("HashingVectorizerEmbeddingConfig", loaded_index.embedder.config)` after `isinstance(loaded_index.embedder.config, HashingVectorizerEmbeddingConfig)` is redundant — drop the cast, mypy already narrowed inside the `if` block; outside the block, hoist the check or use `assert`.
  **Excluded from Pattern A**: anything that's actually a `Mock` / `AsyncMock` / `MagicMock` masquerading as a real type — see Pattern G.
- **Pattern B: ListOfGenericLabels → ListOfLabels.**
  Replace `cast("ListOfLabels", labels)` with:
  ```python
  from tests._helpers import is_strict_labels
  assert is_strict_labels(labels)
  ```
  After the `assert`, `labels` is narrowed to `ListOfLabels`. (Reviewer-verified that this narrowing works under this repo's mypy.)
  Do NOT use the `if not is_strict_labels(labels): pytest.skip(...)` pattern for narrowing: `typing.TypeGuard` (the 3.10-compatible form) only narrows the *positive* branch; the test body after the `if` would still see `ListOfGenericLabels`, not `ListOfLabels`. (That negation-narrowing is `typing.TypeIs`, which requires Python 3.13+ — out of support.) If a test legitimately needs to exercise both variants, do the skip and then re-assert: `if not is_strict_labels(labels): pytest.skip("...")` then `assert is_strict_labels(labels)` immediately after.
- **Pattern C: list[T | None] → list[T] (non-label cases like `list[str | None]`).**
  Replace with an inline runtime check + assert:
  ```python
  assert all(d is not None for d in descriptions)
  ```
  mypy does NOT narrow `list[str | None]` to `list[str]` from a `all(...)` check alone (reviewer-verified), so the call site that consumes the narrowed value may still need an explicit `cast` — in that case KEEP the cast but add the assert above it for runtime safety. Document this in your report.
- **Pattern D: deliberate `cast(Any, ...)` for private-state poking.**
  Keep as-is. Spec-policy escape hatch. Example: `cast("Any", node.module._embedder)` in tests/pipeline/test_inference.py (B8).
- **Pattern E: parametrize Literal narrowing (e.g. `cast("SearchSpacePreset", preset)`, `cast("TaskType", "full_training")`).**
  Keep the cast for now. The underlying issue is a too-narrow `Literal` in src/ or the root conftest — a Phase C decision. Carry forward.
- **Pattern F: wider base-class return type cast (e.g. `cast("tuple[npt.NDArray[Any], list[dict[str, Any]] | None]", scorer.predict_with_metadata(...))`).**
  Default: KEEP the cast. Only replace with a runtime tuple-unpack + assert pattern if all three hold:
  (a) the replacement is **≤3 lines** of asserts,
  (b) no `isinstance` for the same value appears **within the 3 statements immediately above** the cast site (avoids duplicate narrowing),
  (c) at least one downstream operation in the test **actually depends on the runtime type** — indexing (`preds[0]`), attribute access (`preds.shape`), or a follow-up `isinstance`. **A bare `assert preds == <literal>` does not count.**
  Example replacement when all three hold:
  ```python
  preds, meta = scorer.predict_with_metadata(...)
  assert isinstance(preds, np.ndarray)
  assert meta is None or isinstance(meta, list)
  ```
  Otherwise keep the cast. Document the rule you applied in your report. (Tightened gates to drive consistent decisions across subagents.)
- **Pattern G: Mock substitution into a real-type parameter** (e.g. `cast("Generator", client)` where `client = AsyncMock()` or `client = MagicMock()`).
  **KEEP the cast.** Per spec policy item 3: for `unittest.mock.patch` / `MagicMock`, the spec explicitly recommends `cast(Foo, mock)` at the call site rather than widening the test fn signature to `Any`. An `assert isinstance(client, Generator)` would FAIL at runtime since `AsyncMock` is not a `Generator` subclass. The cast is the spec-blessed bridge. Tag these with a one-line `# reason: AsyncMock substituted for Generator dependency` if the cast is currently bare.
- **Pattern H: Scalar coercion from `Any`-typed external sources** (e.g. `cast("Path", importlib.resources.files(...))`, `cast("int", cursor.fetchone()[0])`, `cast("dict[str, Any]", yaml.safe_load(f))`).
  **KEEP the cast.** These come from APIs that return `Any` by design (`yaml.safe_load`, sqlite row tuples, optional accessors). Replacing with `assert isinstance(...)` is technically valid but provides no extra safety the test wouldn't get from a quick smoke assertion. Tag with `# reason: <source> returns Any by API design` if currently bare.

YOUR WORK:
1. `cd` to your worktree root (Agent gave you one).
2. Reset to your existing PR's branch + rebase onto updated b/mypy-on-tests:
   ```bash
   git fetch origin <your-branch> b/mypy-on-tests
   git checkout -B <your-branch> origin/<your-branch>
   git rebase origin/b/mypy-on-tests
   ```
   Expected: clean rebase. Phase B subagent branches only touched `tests/<their-subdir>/` paths; D1 only touched `src/` and `tests/_helpers/` (which was empty before D1 and untouched by any subagent). The path sets are disjoint, so no conflict should arise. If a conflict somehow does appear, stop and report it rather than guessing — the main thread will resolve and resume.
3. Sync deps:
   ```bash
   uv sync --group typing --group test --extra catboost --extra peft --extra transformers --extra sentence-transformers --extra openai
   ```
4. Run mypy on your scope. Note any newly-unused `# type: ignore` (mypy will print them as `[unused-ignore]` errors under `warn_unused_ignores`):
   ```bash
   uv run --group typing --group test mypy <YOUR-SCOPE> 2>&1 | tee /tmp/d2_mypy.txt | tail -30
   ```
5. For each unused ignore, delete the entire `# type: ignore[...] # reason: ...` comment (and the `noqa` if it was paired). Re-run mypy until 0 errors.
6. For each remaining `cast(...)` in your scope, classify per Patterns A–F above and either replace or keep. Document in your report.
7. Final verification:
   ```bash
   uv run --group typing --group test mypy <YOUR-SCOPE>  # exit 0
   uv tool run ruff check <YOUR-SCOPE>                    # exit 0
   uv tool run ruff format --check <YOUR-SCOPE>           # exit 0
   ```
8. Commit (NEW commit on top of your rebased branch — do NOT amend; PR comments may reference the original commits).
   **Substitute your actual subdirectory path for `<YOUR-SUBDIR>` in both the `git add` and the commit message before running** — the literal placeholders must not survive into the commit.
   ```bash
   git add <YOUR-SCOPE>
   git commit -m "$(cat <<'EOF'
test(types): rework <YOUR-SUBDIR> after D1 — drop unused ignores; replace casts with asserts/TypeGuard

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
   ```
9. Force-push to refresh the PR (rebase changed history):
   ```bash
   git push --force-with-lease origin <your-branch>
   ```
   `--force-with-lease` is safer than `--force` — it refuses to overwrite if origin received a new commit you don't have locally.
10. Report and stop. DO NOT wait for CI.

FINAL REPORT FORMAT:
- Rebase: clean / conflict-resolved (with which strategy).
- Mypy exit status (must be 0).
- Unused ignores dropped: count + list (file:line, old code).
- Casts removed: count + list (file:line, old type) and which Pattern (A/B/C/D/E/F) was applied.
- Casts kept: count + list (file:line, type) and which Pattern justifies keeping.
- Frozen-surface change request: any modification you wanted but couldn't make.
- Suspected real `src/` type bug: any new ones (not the 3 already fixed in D1).
- PR URL (same one, now refreshed) + branch name + worktree path.

HARD LIMITS:
- DO NOT modify: `src/`, root `tests/conftest.py`, `tests/_fixtures/`, `tests/_helpers/` (you may *import* from it; the typeguards.py file is now part of the integration surface but you don't modify it), `tests/_transformers/`, `pyproject.toml`, `.github/`, `docs/`.
- DO NOT add dependencies.
- DO NOT refactor for non-typing reasons.
- DO NOT rename test functions or change `parametrize` semantics.
- DO NOT RUN PYTEST LOCALLY.
- If you need a new TypeGuard helper (a recurring pattern across multiple files in your scope that isn't covered by `is_strict_labels`), report it as a frozen-surface change request — do NOT add to `typeguards.py` yourself. Phase C decides whether to add it.

If you cannot complete without violating a hard limit, document in the report and leave the offending cast in place.
```

The main thread then runs the same two-stage review per `superpowers:subagent-driven-development`:
1. Auto-check: diff scope confined to PR's subdir; mypy clean; ruff clean; no new ignores added (only removed).
2. Substantive check: spot-read 3–5 cast removals; confirm `assert isinstance(...)` semantically equivalent to the dropped `cast`.

## Task D2.1 through D2.10: Dispatch subagents for each PR

The 10 subagents are launched in **one message with 10 parallel `Agent` calls**, each instantiating the contract template with:

| Subagent | PR | Branch | Scope | Expected work |
|---|---|---|---|---|
| D2.1 | #313 | `worktree-agent-ad9b2c28651d1c4e3` | `tests/modules/scoring` | Drop ~19 `[arg-type]` ignores (catboost/sklearn `**kwargs` — D1 fixed). KEEP 2 surviving ignores: (a) `test_dnnc.py:43 decimal=0.5` (real pre-existing test bug → Phase C decision), (b) `test_catboost.py features_type=features_type` (test parametrizes literal strings against `features_type: FeaturesType` enum; widening to `FeaturesType | str` is a separate src decision deferred to Phase C). Replace `cast("ListOfLabels", ...)` with `is_strict_labels`; apply Pattern F triple-gate to return-type casts. |
| D2.2 | #304 | `b/mypy-on-tests-modules-decision` | `tests/modules/decision` | 0 ignores to drop. Replace `cast("ListOfLabels", train_labels(0))` with `is_strict_labels`. |
| D2.3 | #307 | `worktree-agent-a61551566a7d532e7` | `tests/modules/{embedding,test_dumper.py,test_regex.py}` | Keep the 1 transformers AutoTokenizer ignore (stub gap, not D1 territory). Replace `isinstance`-guarded casts with asserts. |
| D2.4 | #312 | `b/mypy-on-tests-embedder` | `tests/embedder` | Keep 2 `[unreachable]` ignores (lazy-loading pattern). Replace `cast("...Config", ...)` and `cast("SentenceTransformerEmbeddingBackend", embedder._backend)` (~4 sites) with `assert isinstance(...)` per Pattern A. Replace `cast("ListOfLabels", ...)` with `is_strict_labels`. |
| D2.5 | #306 | `b/mypy-on-tests-data` | `tests/data` | 0 ignores added in B, 0 casts reported. Quick rebase + verify; likely a no-op commit (or a "no rework needed" amendment to the PR title). |
| D2.6 | #311 | `b/mypy-on-tests-generation` | `tests/generation` | Drop 4 `[arg-type]` ignores (Generator(**kwargs) — D1 fixed); keep 1 `[comparison-overlap]` ignore (pre-existing test bug flagged for Phase C). **Keep all ~9 `cast("Generator", client)` calls per Pattern G** — `client` is `AsyncMock()` in the test code, so isinstance narrowing would fail at runtime; the cast is the spec-blessed mock-substitution bridge. If any such cast lacks an inline `# reason:`, add one. |
| D2.7 | #308 | `b/mypy-on-tests-configs` | `tests/configs` | Keep 2 `[arg-type]` ignores (TypeError-assertion tests — intentional). Keep `cast("TaskType", "full_training")` per Pattern E. Keep `cast("dict[str, Any]", ...)` per **Pattern H** (yaml-safe-load returns Any). |
| D2.8 | #310 | `b/mypy-on-tests-pipeline` | `tests/pipeline` | Keep 1 parametrize Literal ignore (Pattern E). Replace `cast("NodeOptimizer", node)` and `cast("...Config", ...)` with asserts per Pattern A. KEEP `cast("Any", node.module._embedder)` per Pattern D. KEEP `cast("Path", ...)` and `cast("int", cursor.fetchone()[0])` per **Pattern H** (importlib.resources + sqlite return Any). |
| D2.9 | #305 | `b/mypy-on-tests-context` | `tests/context` | 0 ignores. Replace `cast("OpenSearchBackend", ...)` and `cast("HashingVectorizerEmbeddingConfig", ...)` with asserts inside the isinstance guards. |
| D2.10 | #309 | `b/mypy-on-tests-misc` | `tests/{callback,ci,metrics}` | Keep 2 `[arg-type]` ignores — they're for the `@ignore_oos` decorator (a separate src/ gap, Phase C decision). |

### Per-task structure (identical shape for D2.1 through D2.10)

- [ ] **Step 1: Dispatch subagent**

Use `Agent({isolation: "worktree", subagent_type: "general-purpose", run_in_background: true, prompt: <contract instantiated for this PR>})`.

- [ ] **Step 2: Review subagent report + wait for refreshed CI**

Per contract above. CI on the refreshed PR will re-run; treat the `test_dump_modules[description_no_llm]` flake per #314 (rerun the job or accept; not blocking).

- [ ] **Step 3: Merge subagent branch into `b/mypy-on-tests` + push to auto-close PR**

```bash
cd /Users/voorhs/repos/lab/AutoIntent/.claude/worktrees/mypy-on-tests
git fetch origin <subagent-branch>
git merge --no-ff origin/<subagent-branch>
git push origin b/mypy-on-tests
```

Merges MUST be serialized (one at a time into `b/mypy-on-tests`).

- [ ] **Step 4: Verify merged state**

```bash
uv run --group typing --group test mypy <PR-SCOPE> 2>&1 | tail -3
```
Expected: `Success`.

---

## Parallelism note (D2)

The 10 D2 subagents dispatch in one message and queue against the runtime concurrency cap (~6–10 parallel slots). Each subagent's work is mypy + ruff only — same memory profile as Phase B (well under the OOM threshold; the prohibition is on pytest).

`tests/_helpers/typeguards.py` is read by potentially multiple subagents (B1/B2/B4 import `is_strict_labels`) but it's read-only for D2 subagents — only the main thread in D1 wrote it. No write contention.

CI minutes cost: 10 PRs each re-running their matrix once. Same trade-off as Phase B — cheap relative to a frozen laptop.

**Orphan CI runs after force-push.** When a D2 subagent `git push --force-with-lease`es, GitHub does not cancel the existing in-flight CI run on the old SHA — that run completes against the old commits and becomes an orphan. A fresh CI run starts against the rebased history. **Treat the post-rebase run as authoritative.** Ignore orphan runs whose head SHA doesn't match the current PR head. Specifically, the Phase B PRs currently running CI at the moment D1 pushes will finish their current run on the old base; the rebased D2 commit triggers a new run. This is normal — flag only if the *latest* run is failing.

**No D1-CI sync gate.** The plan does NOT require D2 subagents to wait for D1's CI (typing.yml against `b/mypy-on-tests`) to complete before rebasing. D1's local mypy verification (Task D1.3) is sufficient pre-flight; remote CI is the post-flight backstop. If D1's CI surfaces an unexpected mypy error, we hot-patch on `b/mypy-on-tests` and the next D2 rebase picks it up. The cost of waiting (10–20 minutes of serialized CI) outweighs the diagnostic benefit.

**Merge ordering: arbitrary.** The 10 D2 subagent diffs touch disjoint test subdirectories by construction (per the spec's Phase B partitioning), so merge order does not affect conflict risk. Merge in whatever order PRs go CI-green. The only sequencing rule is the standard "one merge into `b/mypy-on-tests` at a time" from the parent plan's parallelism note.

---

# Phase D3 — Hand-off back to Phase C

After all D2 PRs are merged into `b/mypy-on-tests` (and origin is pushed), resume the parent plan at:
- `docs/superpowers/plans/2026-06-07-mypy-on-tests.md` → Task **C1** (full mypy across src + tests).

The carry-forward inputs for Phase C are:
- All 10 subagent diffs already merged.
- Remaining ignores (the ones D2 deliberately kept): each carries a code + reason and is auditable in C2.
- Flagged real src/ type bugs (not addressed in D1):
  - `@ignore_oos` decorator signature gap (B10 — `src/autointent/metrics/retrieval.py` ~line 117).
  - `root tests/conftest.TaskType` Literal missing `"full_training"` (B7).
  - `src/autointent/utils.py:load_search_space` return-type narrowness for yaml-as-dict variants (B7).
- Flagged latent test bugs:
  - `test_basic_synthesizer.py:21-23` (B6) — always-true assertion, copy/paste bug.
  - `test_dnnc.py:43` (B1) — `decimal=0.5` where int required (Phase C decision: fix or accept the ignore).
  - `test_check_split_readiness.py:154` (B5) — already fixed inline by B5.
  - `test_vector_index.py:156` (B9) — already fixed inline by B9.

No changes to Phase C's tasks themselves. Phase C still: full mypy → ignore audit → src/ bug escalation → flip gate (remove `continue-on-error`) → push branch + open PR to `dev` (no merge).

---

# Self-review

## Spec coverage

This plan extends (does not replace) the parent plan. Its goals come from Phase B report consolidation, not the original spec, so spec-coverage doesn't apply. Cross-check:

| Phase B report finding | D1/D2 addressing it |
|---|---|
| B1: ~20 catboost/sklearn `[arg-type]` ignores from kwargs bug | D1.1 + D2.1 (drop) |
| B6: 4 generator `[arg-type]` ignores from same kwargs bug | D1.1 + D2.6 (drop) |
| B1, B2, B4: `cast("ListOfLabels", ...)` recurring pattern | D1.2 (`is_strict_labels`) + D2.{1,2,4} (replace) |
| B3, B4, B8, B9: subclass `cast` after isinstance guard | D2 contract Pattern A (assert isinstance) |
| B8: deliberate `cast(Any, ...)` for private state | D2 contract Pattern D (keep) |
| B7: `cast("TaskType", "full_training")` (literal incomplete) | D2 contract Pattern E (keep, Phase C decides widening) |
| B1: return-type casts from base class | D2 contract Pattern F (default keep; replace only if ≤3 lines + non-redundant + already exercised) |
| B6: `cast("Generator", AsyncMock())` mock substitution | D2 contract Pattern G (keep per spec policy item 3) |
| B7/B8: `cast` from `Any`-returning external sources (yaml, sqlite, importlib.resources) | D2 contract Pattern H (keep, tag with reason) |
| B6/B1: Generator/Sklearn/CatBoost src kwargs bugs | D1.1 (4-signature fix across 3 files; reviewer caught missed Sklearn `from_context`) |
| @ignore_oos decorator gap (B10) | NOT in D1; Phase C step 3 |
| `tests/conftest.TaskType` widening | NOT in D1 (frozen); Phase C step 3 |
| `load_search_space` return type narrowness | NOT in D1; Phase C step 3 |

## Placeholder scan

- No "TBD", "TODO", or "implement later". Each step has the actual command or code.
- The `<YOUR-SCOPE>` and `<your-branch>` placeholders in the D2 contract template are filled per-subagent in the dispatch table — that's the template substitution mechanism, not a planning placeholder.

## Type consistency

- `is_strict_labels` is defined once in `tests/_helpers/typeguards.py`, re-exported from `tests/_helpers/__init__.py`, and referenced consistently across D2.1, D2.2, D2.4 in the dispatch table.
- Branch names in the dispatch table match the ones reported by the Phase B subagents (some are `b/mypy-on-tests-<scope>` from explicit `git push -u`, some are `worktree-agent-<id>` from harness auto-naming). Both are valid origin branches and both PRs are already open.
- PR numbers (304–313) match the gh-list snapshot.

## Notes / known sharp edges

- **The D1 commit modifies `tests/_helpers/__init__.py`.** This file was implicitly frozen in Phase A; D1 deliberately unfreezes it for one commit to add the new `is_strict_labels` export. This is the only D1 amendment to the parent plan's freeze list.
- **Rebase + force-push to a PR can confuse reviewers** if the PR has existing review comments anchored to old line numbers. None of the 10 PRs has external review comments yet (they're internal subagent work-in-progress), so this is fine here. For future PRs with review history, consider amend-with-merge-commit instead.
- **Pattern F is the riskiest replacement.** Some return-type casts are honestly easier to keep than to expand into 3 lines of runtime checks. The contract explicitly tells subagents this is a judgment call and to document which way they went. Phase C re-reviews.
- **CI cost.** 10 PRs each get one extra round of CI (same matrix as Phase B). On the order of a few hundred CI-minutes total. Cheap, but not zero.
