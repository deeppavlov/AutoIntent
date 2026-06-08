# Strict mypy on `tests/` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable `mypy --strict` over `tests/` in CI and fix all 745 existing errors, so test code drifting from typed `src/` APIs (the #296 class of bugs) fails type-check before merge.

**Architecture:** Three phases on a single branch `b/mypy-on-tests`. Phase A (sequential, main thread) lands the CI config in warn-only mode and types the shared test surface (root `conftest.py`, `_fixtures/`, `_transformers/`). Phase B (parallel, 10 fan-out subagents) — each subagent owns one test subdirectory and reduces it to zero mypy errors. Phase C (sequential, main thread) integrates, verifies, flips the CI gate from warn-only to enforced, opens PR (no merge).

**Tech Stack:** Python 3.10+, mypy 1.x (strict), pydantic v2 mypy plugin, pytest 8.x, uv 0.10 for environment + worktree management, git worktrees for subagent isolation.

**Spec:** `docs/superpowers/specs/2026-06-07-mypy-on-tests-design.md` — read this BEFORE starting any task. Especially: the "Policy for hard-to-type pytest patterns" section governs every type fix.

---

## File Structure

Tasks in this plan touch the following files (created or modified):

| Path | Touched by | Responsibility |
|---|---|---|
| `.github/workflows/typing.yml` | A1, C4 | mypy CI invocation; warn-only flag |
| `pyproject.toml` | A1 | mypy overrides for `tests.server.*` + `ignore_missing_imports` additions |
| `tests/conftest.py` | A2 | Root fixtures, importable from every subdir |
| `tests/_fixtures/**/*.py` | A3 | Shared mock generators, container helpers |
| `tests/_transformers/**/*.py` | A4 | Test-only transformer mocks |
| `tests/modules/scoring/**` | B1 | Scoring module tests + scoring-private conftest if any |
| `tests/modules/decision/**` | B2 | Decision module tests + `decision/conftest.py` |
| `tests/modules/{embedding,test_dumper.py,test_regex.py}` | B3 | Rest of modules tests |
| `tests/embedder/**` | B4 | Embedder tests + `embedder/conftest.py` |
| `tests/data/**` | B5 | Dataset tests |
| `tests/generation/**` | B6 | Generation tests |
| `tests/configs/**` | B7 | Config tests |
| `tests/pipeline/**` | B8 | Pipeline tests |
| `tests/context/**` | B9 | Context tests |
| `tests/{callback,ci,metrics}/**` | B10 | Misc subdirs + `ci/conftest.py` |

Files NOT touched by this plan: anything under `src/` (out-of-scope by spec; exceptions logged in subagent reports for Phase C decision); `tests/_helpers/` (already 0 errors, locked); `tests/server/**` (excluded via `tests.server.*` override).

---

## Conventions

- **All commands run from the worktree root**: `/Users/voorhs/repos/lab/AutoIntent/.claude/worktrees/mypy-on-tests` (or for Phase B subagents: their own worktree root).
- **Maximal install for Phase A / Phase B**: every uv sync uses `uv sync --group typing --extra catboost --extra peft --extra transformers --extra sentence-transformers --extra openai`. This installs all extras tests can import; uv's global cache amortizes downloads across worktrees. This is heavier than each subagent's strict minimum but eliminates a class of "missing optional extra" failures and matches the existing typing.yml workflow's pattern.
- **Spec is authoritative**: when this plan and the spec disagree, the spec wins; flag the inconsistency in your report.

---

# Phase A — Main thread, sequential

## Task A1: Land mypy-on-tests CI infrastructure in warn-only mode

**Files:**
- Modify: `.github/workflows/typing.yml` (the `Run mypy` step)
- Modify: `pyproject.toml` (lines 275–298 for `ignore_missing_imports`; add new `[[tool.mypy.overrides]]` block for `tests.server.*`)

- [ ] **Step 1: Read the spec sections for config changes**

Read `docs/superpowers/specs/2026-06-07-mypy-on-tests-design.md` sections "Configuration changes" and "Phase A — Main thread, sequential" step 4.

- [ ] **Step 2: Edit `.github/workflows/typing.yml`**

Find the `Run mypy` step. Modify it to include `tests` in the mypy invocation and add `continue-on-error: true` for warn-only mode:

```yaml
      - name: Run mypy
        continue-on-error: true
        run: uv run mypy src/autointent tests
```

- [ ] **Step 3: Edit `pyproject.toml` — extend `ignore_missing_imports` list**

Find the existing `[[tool.mypy.overrides]]` block at line 275 (the one with `ignore_missing_imports = true`). Add two entries to its `module = [...]` list:

```toml
    "testcontainers.opensearch",
    "warm_hf_cache",
```

Place them alphabetically among the existing entries.

- [ ] **Step 4: Edit `pyproject.toml` — add `tests.server.*` override**

After the last existing `[[tool.mypy.overrides]]` block (the one with `ignore_errors = true` for `autointent.server.*` around line 307), append a new block:

```toml
[[tool.mypy.overrides]]
module = ["tests.server.*"]
ignore_errors = true
```

- [ ] **Step 5: Verify mypy still parses the config**

Run:
```bash
cd /Users/voorhs/repos/lab/AutoIntent/.claude/worktrees/mypy-on-tests
uv sync --group typing --extra catboost --extra peft --extra transformers --extra sentence-transformers --extra openai
uv run --group typing mypy --help > /dev/null && echo "config parses"
```
Expected: `config parses`. Any TOML/config error fails this task.

- [ ] **Step 6: Confirm `tests.server.*` exemption works**

Run:
```bash
uv run --group typing mypy tests/server 2>&1 | tail -3
```
Expected: `Success: no issues found in N source files` (the override silences all errors).

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/typing.yml pyproject.toml
git commit -m "ci(typing): include tests/ in mypy (warn-only); add tests.server.* + missing-import overrides

Warn-only mode (continue-on-error) protects b/mypy-on-tests during Phase B
subagent fan-out. The gate flips to enforced in Phase C after all subdirs
land. testcontainers.opensearch + warm_hf_cache are the only baseline
[import-untyped]/[import-not-found] cases.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task A2: Type `tests/conftest.py` (38 errors)

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Read current state**

```bash
uv run --group typing mypy tests/conftest.py 2>&1 | head -50
```
Capture the full error list. Common patterns: `[no-untyped-def]` on helper functions, `[arg-type]` on calls into `autointent` APIs, `[no-untyped-call]` on calls into untyped helpers within tests.

- [ ] **Step 2: Apply type annotations per spec policy**

For each error in the file, apply a fix following the spec's "Policy for hard-to-type pytest patterns" section:
- **Functions**: add full signatures including return type. Use `-> None` for void functions; use `pytest`'s typed fixture protocols (`Path`, `MonkeyPatch`, `LogCaptureFixture`) when consuming pytest fixtures.
- **Yield-style fixtures**: return type `Iterator[T]` from `collections.abc`.
- **Pydantic `**kwargs` spread** (spec policy item 5): if present, prefer enumerating fields; otherwise `cast(...)` or `model_construct`.
- **`get_search_space`, `get_dataset_path`, `setup_environment`** (visible at top of file): annotate their actual return types (`SearchSpace`, `Path`, `Path` respectively — verify from source).
- **`_disable_transformers_mistral_regex_patch`**: annotate as `-> None`, annotate inner `_noop_patch_mistral_regex` parameters. The `cls`/`tokenizer`/`*args`/`**kwargs` form is tricky — likely needs `cast(Any, ...)` or `# type: ignore[no-untyped-def]` with reason "monkey-patched into transformers internals".

This file is the foundation other subagents depend on, so prefer **enumerating exact types** over `Any`.

- [ ] **Step 3: Verify zero errors**

```bash
uv run --group typing mypy tests/conftest.py 2>&1 | tail -3
```
Expected: `Success: no issues found in 1 source file`.

- [ ] **Step 4: Verify collection still works**

```bash
uv run pytest --collect-only tests/conftest.py 2>&1 | tail -5
```
Expected: clean exit (warnings ok, errors not).

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py
git commit -m "test(types): annotate root conftest.py (38→0 mypy errors)

Frozen shared surface for Phase B. Subagents will type-annotate their
per-subdir fixtures consuming this surface.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task A3: Type `tests/_fixtures/**` (10 errors)

**Files:**
- Modify: every `.py` file under `tests/_fixtures/` with mypy errors

- [ ] **Step 1: Capture baseline**

```bash
uv run --group typing mypy tests/_fixtures 2>&1 | tee /tmp/A3_baseline.txt | tail -3
```
Expected: `Found 10 errors`. Read the file to understand each error's context.

- [ ] **Step 2: Apply fixes per spec policy**

Walk the error list. Known patterns in this directory:
- `tests/_fixtures/mock_generator.py` patches `autointent.modules.scoring._description.llm_encoder` — a private module attribute. Use `cast(Any, ...)` at the patch site with `# reason:` comment (escape hatch per spec policy item 4). Document in the subagent report that this is the canonical example reviewers should look at in Phase C.
- `tests/_fixtures/opensearch_container.py` already triggers `[import-untyped]` from `testcontainers.opensearch` — Task A1 added this to `ignore_missing_imports`, so this specific error vanishes; confirm.
- `tests/_fixtures/fake_openai_embedding.py` / `respx_openai.py` — likely missing annotations on helper functions and on what they return. Use `respx`-typed return values where available.

- [ ] **Step 3: Verify zero errors**

```bash
uv run --group typing mypy tests/_fixtures 2>&1 | tail -3
```
Expected: `Success: no issues found in N source files`.

- [ ] **Step 4: Verify imports still resolve**

```bash
uv run pytest --collect-only tests/_fixtures 2>&1 | tail -5
```
(Or, since `_fixtures/` isn't a test directory itself, smoke-check by collecting any test that depends on these fixtures, e.g., `uv run pytest --collect-only tests/data 2>&1 | tail -3`.)
Expected: no import errors.

- [ ] **Step 5: Commit**

```bash
git add tests/_fixtures/
git commit -m "test(types): annotate tests/_fixtures (10→0 mypy errors)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task A4: Type `tests/_transformers/**` (19 errors)

**Files:**
- Modify: every `.py` file under `tests/_transformers/` with mypy errors

- [ ] **Step 1: Capture baseline**

```bash
uv run --group typing mypy tests/_transformers 2>&1 | tee /tmp/A4_baseline.txt | tail -3
```
Expected: `Found 19 errors`.

- [ ] **Step 2: Apply fixes per spec policy**

This directory holds test-only transformer mocks. Common patterns:
- `BatchEncoding` and `PreTrainedTokenizer*` return types from `transformers` — use the typed surfaces (`transformers` has `py.typed`, so types resolve directly).
- `__call__` signatures on mock tokenizer classes — these may be the bulk of `[no-untyped-def]` errors. Use `BatchEncoding` as the return type and `str | list[str] | ...` for inputs.

- [ ] **Step 3: Verify zero errors**

```bash
uv run --group typing mypy tests/_transformers 2>&1 | tail -3
```
Expected: `Success`.

- [ ] **Step 4: Verify imports**

```bash
uv run pytest --collect-only tests/_transformers 2>&1 | tail -5
```
Expected: no import errors.

- [ ] **Step 5: Commit**

```bash
git add tests/_transformers/
git commit -m "test(types): annotate tests/_transformers (19→0 mypy errors)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task A5: Verify Phase A shared surface is clean

**Files:** none modified.

- [ ] **Step 1: Run mypy on shared surface only**

```bash
uv run --group typing mypy src/autointent tests/conftest.py tests/_fixtures tests/_helpers tests/_transformers 2>&1 | tail -3
```
Expected: `Success: no issues found in N source files`.

- [ ] **Step 2: Tag the commit for Phase B subagents to fork from**

```bash
git tag phase-A-shared-surface
```

This tag is the explicit fork point for Phase B subagent worktrees. If a subagent's worktree creation needs an explicit ref, use `phase-A-shared-surface`.

- [ ] **Step 3: Verify branch state**

```bash
git log --oneline -8
git status
```
Expected: clean working tree; 5 commits since branching from `dev` (spec, plan, infra, conftest+_fixtures or split, _transformers).

If `Step 1` fails: stop, do not proceed to Phase B. Investigate and re-do whichever A-task introduced the regression.

- [ ] **Step 4: Push `b/mypy-on-tests` to origin so Phase B PRs have a target**

```bash
git push -u origin b/mypy-on-tests
git push origin phase-A-shared-surface  # tag, optional but useful as a sanity anchor
```

Phase B subagents open PRs targeting this branch on GitHub so the `ci.yaml` and `typing.yml` workflows run pytest + mypy on each subagent's diff. **Without this push, the subagent PRs would have nowhere to target, and verification would have to fall back to local pytest — which is exactly what caused the previous OOM** when 10 parallel subagents each loaded the full ML stack (torch / transformers / sentence-transformers) into memory at once.

Note: the on-branch CI runs are bounded by the `Typing` workflow already having `continue-on-error: true` from Commit 3, so a mid-flight push won't paint the branch red on the typing check.

---

# Phase B — Subagent fan-out, parallel

All Phase B tasks share the same shape (subagent contract from the spec). The variable parts are the subdir, baseline error count, and worktree branch name. Each subagent is dispatched from the main thread in a single `Agent({isolation: "worktree"})` call.

## Subagent contract template (used by every B-task)

When dispatching a B-task subagent, the main thread sends a prompt structured as:

```
You are fixing mypy errors in `tests/<SUBDIR>` (baseline: <N> errors).

REQUIRED READING (do this first, in order):
1. `docs/superpowers/specs/2026-06-07-mypy-on-tests-design.md` — the spec.
2. `docs/superpowers/plans/2026-06-07-mypy-on-tests.md` — this plan, specifically your task and the "Policy for hard-to-type pytest patterns" recap below.

POLICY RECAP (full text in spec §"Policy for hard-to-type pytest patterns"):
- Annotate fixtures with return types; `Iterator[T]` for yield-style.
- Keep parametrize argvalues homogeneous; split parametrize blocks for mixed types.
- `cast(Foo, mock)` or annotate `MagicMock` at the call site; do NOT widen test signatures.
- Escape hatch: `# type: ignore[<code>]` with `# reason:` comment. Multi-code form allowed.
- Pydantic `**kwargs` spread: enumerate fields, or use `model_construct(**kwargs)` with cast.
- For `pytest.skip` + unreachable: prefer `pytest.importorskip` or `pytest.mark.skipif`.

YOUR WORK:
1. Confirm baseline: `uv run --group typing mypy tests/<SUBDIR> 2>&1 | tail -3` shows ~<N> errors.
2. Fix every error per the policy. Touch ONLY files inside `tests/<SUBDIR>` (and your subdir's own `conftest.py` if present).
3. Verify mypy: `uv run --group typing mypy tests/<SUBDIR>` exits with zero errors.
4. Commit with message:
   `test(types): annotate tests/<SUBDIR> (<N>→0 mypy errors)\n\nCo-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`
5. Push your worktree's branch to origin: `git push -u origin <your-branch-name>`.
6. Open a PR against `b/mypy-on-tests` (the parent integration branch, already pushed by Phase A Task A5):
   ```bash
   gh pr create --base b/mypy-on-tests --head <your-branch-name> \
     --title "test(types): annotate tests/<SUBDIR> (<N>→0 mypy errors)" \
     --body "Subagent diff for the strict-mypy-on-tests effort. See docs/superpowers/specs/2026-06-07-mypy-on-tests-design.md. CI on this PR is the verification surface — pytest + typing run on GitHub Actions, not locally."
   ```
   This triggers `ci.yaml` (full pytest matrix on the diff) and `typing.yml`.
7. Report (final message to main thread):
   - Mypy exit status (must be 0).
   - PR URL.
   - Number of `# type: ignore` added; list each with file:line, code, reason.
   - Any frozen-surface change request (file you couldn't fix without modifying frozen code).
   - Any suspected real `src/` type bug discovered (with file:line and call site).
   - Do NOT wait for CI to finish before reporting. The main thread polls `gh pr checks` and gates the merge on CI green.

HARD LIMITS:
- **DO NOT run pytest locally** — not even `pytest --collect-only`. Earlier concurrent local pytest invocations across 10 parallel subagents OOM'd the host because each one loaded the full ML stack (torch, transformers, sentence-transformers) into memory. Verification happens on GitHub Actions via your PR. Mypy alone is fine to run locally (lightweight).
- Do not modify: `src/`, root `tests/conftest.py`, `tests/_fixtures/`, `tests/_helpers/`, `tests/_transformers/`, `pyproject.toml`, `.github/`, `docs/`. Your subdir's own `conftest.py` is in-scope.
- Do not add dependencies.
- Do not refactor for non-typing reasons.
- Do not rename test functions or change parametrize semantics.

If you cannot reach 0 errors without violating a hard limit, apply a scoped
`# type: ignore[<code>] # reason: <explanation>` and document it in your report
so the main thread can decide whether to widen a frozen helper or accept the ignore.
```

The main thread then performs the two-stage review per `superpowers:subagent-driven-development`:
1. Auto-check: diff scope confined to `tests/<SUBDIR>`; mypy clean; ignores carry codes + reasons.
2. Substantive check: spot-read 3–5 diff chunks for behavior changes (mock substitutions, fixture narrowing).

If checks pass, merge the subagent's branch into `b/mypy-on-tests`. If checks fail, send back to the subagent with specific corrections.

---

## Task B1: `tests/modules/scoring` (153 errors)

**Files:** `tests/modules/scoring/**/*.py`

- [ ] **Step 1: Dispatch subagent**

From the main thread on `b/mypy-on-tests` at `phase-A-shared-surface` (or later), dispatch:

```
Agent({
  description: "mypy fixes: tests/modules/scoring",
  subagent_type: "general-purpose",
  isolation: "worktree",
  prompt: <subagent contract template with SUBDIR=modules/scoring, N=153>
})
```

- [ ] **Step 2: Review subagent report + wait for the PR's CI**

Verify from the report:
- Mypy exit 0 on `tests/modules/scoring` ✓
- A PR URL targeting `b/mypy-on-tests` was provided ✓
- All `# type: ignore` have codes + reasons ✓
- No frozen-surface modifications ✓

Then poll CI on the subagent's PR:
```bash
gh pr checks <PR-URL>
```
- `ci.yaml` (full pytest matrix on the diff): **must pass before merging** — this is the real verification that replaces what used to be local `pytest --collect-only`.
- `typing.yml`: informational on this branch (still warn-only via Phase A Commit 3); merge is not gated on it. Phase C flips it to enforced after all subagents land.

If `ci.yaml` fails, do NOT merge. Send the subagent back with the failing test output (pasted from `gh pr view <PR-URL> --json statusCheckRollup`). If CI is still running, you can review other subagents' reports in parallel — their PRs run independently on GitHub.

- [ ] **Step 3: Merge subagent branch into `b/mypy-on-tests` + push to auto-close the PR**

```bash
cd /Users/voorhs/repos/lab/AutoIntent/.claude/worktrees/mypy-on-tests
git fetch origin <subagent-branch>
git merge --no-ff origin/<subagent-branch>
# Alternative if the subagent's worktree is locally accessible (cherry-pick form):
# git -C <subagent-worktree> format-patch -1 --stdout | git am
git push origin b/mypy-on-tests
```

The final `git push` lands the subagent's commit on the remote target branch, which auto-closes the subagent's PR (GitHub detects the commits are now in the target).

- [ ] **Step 4: Verify the merged state**

```bash
uv run --group typing mypy tests/modules/scoring 2>&1 | tail -3
```
Expected: `Success`.

---

## Task B2: `tests/modules/decision` (51 errors, includes `decision/conftest.py`)

**Files:** `tests/modules/decision/**/*.py` (including `tests/modules/decision/conftest.py`)

- [ ] **Step 1: Dispatch subagent**

```
Agent({
  description: "mypy fixes: tests/modules/decision",
  subagent_type: "general-purpose",
  isolation: "worktree",
  prompt: <subagent contract template with SUBDIR=modules/decision, N=51>
})
```

Note in the prompt: this subdir has its own `conftest.py` (fixtures: `multiclass_fit_data`, `multilabel_fit_data`, `scores`). The subagent owns it.

- [ ] **Step 2: Review subagent report.** (Same checklist as B1 Step 2.)

- [ ] **Step 3: Merge subagent branch.** (Same as B1 Step 3.)

- [ ] **Step 4: Verify merged state**

```bash
uv run --group typing mypy tests/modules/decision 2>&1 | tail -3
```
Expected: `Success`.

---

## Task B3: `tests/modules/{embedding,test_dumper.py,test_regex.py}` (45 errors)

**Files:** `tests/modules/embedding/**`, `tests/modules/test_dumper.py`, `tests/modules/test_regex.py`

- [ ] **Step 1: Dispatch subagent**

```
Agent({
  description: "mypy fixes: tests/modules/{embedding,test_dumper,test_regex}",
  subagent_type: "general-purpose",
  isolation: "worktree",
  prompt: <subagent contract template; SUBDIR is described as the three paths;
          N=45 (19 in embedding + 21 in test_dumper.py + 5 in test_regex.py);
          mypy command: `uv run --group typing mypy tests/modules/embedding tests/modules/test_dumper.py tests/modules/test_regex.py`>
})
```

- [ ] **Step 2–4:** Same shape as B1.

---

## Task B4: `tests/embedder` (108 errors, includes `embedder/conftest.py`)

**Files:** `tests/embedder/**/*.py`

- [ ] **Step 1: Dispatch subagent** with SUBDIR=embedder, N=108. Note that `tests/embedder/conftest.py` is subagent-owned and exports `vllm_available` and config fixtures — `pytest.importorskip("vllm")` pattern is in scope here.

- [ ] **Step 2–4:** Same shape as B1.

---

## Task B5: `tests/data` (78 errors)

**Files:** `tests/data/**/*.py`

- [ ] **Step 1: Dispatch subagent** with SUBDIR=data, N=78.

- [ ] **Step 2–4:** Same shape as B1.

---

## Task B6: `tests/generation` (65 errors)

**Files:** `tests/generation/**/*.py`

- [ ] **Step 1: Dispatch subagent** with SUBDIR=generation, N=65.

- [ ] **Step 2–4:** Same shape as B1.

---

## Task B7: `tests/configs` (60 errors)

**Files:** `tests/configs/**/*.py`

- [ ] **Step 1: Dispatch subagent** with SUBDIR=configs, N=60. Pydantic plugin friction is most likely here — flag this subagent to follow spec policy item 5 carefully.

- [ ] **Step 2–4:** Same shape as B1.

---

## Task B8: `tests/pipeline` (41 errors)

**Files:** `tests/pipeline/**/*.py`

- [ ] **Step 1: Dispatch subagent** with SUBDIR=pipeline, N=41.

- [ ] **Step 2–4:** Same shape as B1.

---

## Task B9: `tests/context` (24 errors)

**Files:** `tests/context/**/*.py`

- [ ] **Step 1: Dispatch subagent** with SUBDIR=context, N=24.

- [ ] **Step 2–4:** Same shape as B1.

---

## Task B10: `tests/{callback,ci,metrics}` (34 errors, includes `ci/conftest.py`)

**Files:** `tests/callback/**`, `tests/ci/**`, `tests/metrics/**`

- [ ] **Step 1: Dispatch subagent** with SUBDIR described as the three paths, N=34 (7+10+17). `tests/ci/conftest.py` is subagent-owned. The mypy command in the contract:
  `uv run --group typing mypy tests/callback tests/ci tests/metrics`.

- [ ] **Step 2–4:** Same shape as B1.

---

## Parallelism note

B1–B10 are independent and can be launched in a single message with 10 parallel `Agent` calls. The two-stage reviews (Step 2 of each task) happen as subagent reports come back. The branch merges (Step 3 of each task) MUST be serialized — only one merge into `b/mypy-on-tests` at a time. Cherry-picking the patches in any order is fine because their file sets are disjoint, so no merge conflicts are expected.

If two subagents both report a request to widen the same frozen helper (e.g., both want `Dataset` widened to `Dataset | None`), Phase C decides; do not block the merges on this — accept their local `cast()` workarounds and revisit centrally.

**Host-OOM safety (why subagents push + open PRs instead of running pytest locally)**: an earlier execution attempt of this plan froze the user's laptop because 10 parallel subagents simultaneously invoked `pytest --collect-only`, each loading the full ML import surface (torch / transformers / sentence-transformers) into memory. The verification step has been moved to GitHub Actions: each subagent pushes its branch and opens a PR against `b/mypy-on-tests`, and `ci.yaml` runs the real pytest matrix on GitHub. Subagents run only mypy locally, which is light enough that 10 parallel processes are not a memory concern. Trade-off: 10 simultaneous CI runs cost some GitHub Actions minutes but cost zero host RAM, which is the right side of the trade to be on.

---

# Phase C — Main thread, sequential

## Task C1: Full mypy across src + tests

**Files:** none modified (read-only check).

- [ ] **Step 1: Run full mypy**

```bash
cd /Users/voorhs/repos/lab/AutoIntent/.claude/worktrees/mypy-on-tests
uv run --group typing mypy src/autointent tests 2>&1 | tail -10
```
Expected: `Success: no issues found in N source files`.

- [ ] **Step 2: If any errors remain, identify cross-subdir issues**

Likely causes:
- A subagent's local `cast()` widened a return type that another subagent narrowed.
- A `# type: ignore` became unused after another subagent's fix (warn_unused_ignores flagged it).

Fix each issue with a minimal commit. If a frozen helper needs widening:

```bash
# Example: widen tests/_fixtures/foo.py return type from Dataset to Dataset | None
git add tests/_fixtures/foo.py
git commit -m "test(types): widen <helper> return type to satisfy subagents B<N> and B<M>

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

- [ ] **Step 3: Re-run mypy and confirm clean**

```bash
uv run --group typing mypy src/autointent tests 2>&1 | tail -3
```
Expected: `Success`.

---

## Task C2: Review all `# type: ignore` additions

**Files:** read-only inspection.

- [ ] **Step 1: List all ignores added on this branch**

```bash
git diff origin/dev -- tests/ | grep -nE '^\+.*# type: ignore' | head -60
```

- [ ] **Step 2: Validate each ignore against policy**

For each ignore added, confirm:
- It carries a specific error code in `[brackets]`, not bare.
- It has a `# reason: ...` comment on the same line or the line above.
- The reason is concrete (not "mypy complains").

- [ ] **Step 3: Reject lazy ignores**

If any ignore fails the check, edit the file to fix the annotation properly OR write a better reason comment, then re-run mypy (Task C1 Step 3). Commit any fixes:

```bash
git add <files>
git commit -m "test(types): tighten type:ignore comments per spec policy

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task C3: Decide on real src/ type bugs flagged by subagents

**Files:** depends on what subagents flagged.

- [ ] **Step 1: Aggregate subagent reports**

Collect every "suspected real src/ type bug" from B1–B10 reports. Group by `src/` file.

- [ ] **Step 2: For each suspected bug**

Decide:
- **Fix here**: if the fix is one-line and localized to `src/`, apply it in a focused commit:
  ```bash
  git add src/autointent/<file>
  git commit -m "fix(src): <one-line type fix> (revealed by tests during mypy-on-tests)

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
  ```
  Then remove the subagent's local `cast()` workaround in a follow-up commit.
- **Defer to issue**: if the fix is non-trivial or touches behavior, open a GitHub issue (`gh issue create`) describing the bug, and leave the subagent's `cast()` workaround in place with the issue link added to its reason comment.

- [ ] **Step 3: Re-run full mypy**

```bash
uv run --group typing mypy src/autointent tests 2>&1 | tail -3
```
Expected: `Success`.

---

## Task C4: Flip the CI gate from warn-only to enforced

**Files:**
- Modify: `.github/workflows/typing.yml` (remove `continue-on-error: true`)

- [ ] **Step 1: Edit `.github/workflows/typing.yml`**

Remove the `continue-on-error: true` line from the `Run mypy` step. Final state:

```yaml
      - name: Run mypy
        run: uv run mypy src/autointent tests
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/typing.yml
git commit -m "ci(typing): enforce mypy on tests/ (remove warn-only)

All test subdirs now clean. mypy regressions on test code will fail
the Typing job, catching #296-class drift at PR time.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

- [ ] **Step 3: Local sanity run of the same command CI will run**

```bash
uv sync --group typing --extra catboost --extra peft --extra transformers --extra sentence-transformers --extra openai
uv run mypy src/autointent tests 2>&1 | tail -3
```
Expected: `Success`. (No `--group` flag needed in the actual run because `uv sync` already populated the venv; matches CI's invocation.)

---

## Task C5: Push branch and open PR (do not merge)

**Files:** none modified.

- [ ] **Step 1: Push the branch**

```bash
git push -u origin b/mypy-on-tests
```

- [ ] **Step 2: Open PR against `dev`**

```bash
gh pr create --base dev --head b/mypy-on-tests --title "ci: enforce mypy --strict on tests/ (closes #296-class drift)" --body "$(cat <<'EOF'
## Summary
- Extend the Typing CI job to run mypy --strict over tests/ in addition to src/autointent.
- Fix all 745 baseline mypy errors across tests/ (10 subagent-authored subdir commits + Phase A shared-surface).
- Add tests.server.* exemption (mirrors src/autointent.server.* policy) and ignore_missing_imports for testcontainers.opensearch + warm_hf_cache.
- Final commit removes warn-only mode; the gate is enforced.

## Motivation
Test code drifts against src/ APIs silently when no CI step exercises it under typing rules. Issue #296 is a textbook case: Pipeline.fit(sampler=...) was removed but a test kept using the kwarg for weeks, undetected because the file was orphaned from the pytest matrix. Strict mypy on tests catches this class of bug at PR time.

## Design
Full design: docs/superpowers/specs/2026-06-07-mypy-on-tests-design.md
Implementation plan: docs/superpowers/plans/2026-06-07-mypy-on-tests.md

## Test plan
- [ ] Typing CI job is green with the gate enforced.
- [ ] All existing pytest jobs remain green.
- [ ] Every # type: ignore added carries a specific code and an inline reason (spot-check during review).
- [ ] No new dependencies in pyproject.toml.

## Do not merge
Per spec §"Phase C — Main thread, sequential" step 7 and user request: this PR is opened for review, not auto-merged. The user will merge manually after review.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Capture PR URL**

Save the PR URL. Report it to the user along with: number of commits on the branch, number of `# type: ignore` introduced, any deferred-to-issue src/ bugs.

- [ ] **Step 4: STOP**

Do **not** merge the PR. The user reviews and merges manually.

---

# Self-review

## Spec coverage

| Spec section | Implementing task(s) |
|---|---|
| Goal — strict mypy on tests/ in CI | A1, C4 |
| Scope — 745 errors fixed | A2–A4, B1–B10 |
| Scope — minimum config overrides | A1 |
| Configuration: typing.yml | A1, C4 |
| Configuration: tests.server.* override | A1 |
| Configuration: ignore_missing_imports additions | A1 |
| Policy: all 7 items | Subagent contract template (re-stated for each B-task); A2–A4 inherit by reading spec |
| Phase A: 4 commits (spec, plan, infra, shared-surface) | spec = previous commit; plan = THIS commit; A1 = infra; A2–A4 = shared-surface (3 commits or 1) |
| Phase B: 10 subagents | B1–B10 |
| Phase B: per-subdir conftest ownership | Restated in B2, B4, B10 |
| Phase C: integrate + gate flip | C1, C4 |
| Phase C: review ignores | C2 |
| Phase C: src/ bug escalation | C3 |
| Done criteria — mypy clean | C1, C3 Step 3 |
| Done criteria — pytest green | Phase B subagent PRs' `ci.yaml` runs + final PR CI |
| Done criteria — gate enforced | C4 |
| Done criteria — no new deps | A1 verification |
| Done criteria — every ignore has code+reason | C2 |
| Done criteria — PR open, not merged | C5 Step 4 |

## Placeholder scan

No "TBD", "TODO", or "implement later" present. Every task has concrete steps and verification commands.

## Type consistency

- Branch name `b/mypy-on-tests` matches the spec and is consistent across A, B, C.
- Tag name `phase-A-shared-surface` introduced in A5 and referenced in B1's worktree fork point.
- Subagent contract template re-used identically across B1–B10 (with variable SUBDIR/N).
- File paths consistent: `tests/conftest.py` (root, frozen) vs `tests/<subdir>/conftest.py` (subagent-owned).

## Notes / known sharp edges

- **A2/A3/A4 may merge into a single commit** if the changes are small and atomic. The plan presents them as three commits for clarity, but Phase A producing one combined "shared-surface" commit (the spec's wording) is acceptable. If combined, update the commit message accordingly and skip the per-task commit step. Phase A5 verification is the gate; the number of sub-commits doesn't matter.
- **Phase B parallelism**: dispatching 10 subagents in one message uses 10 worktrees simultaneously. The runtime concurrency cap (`min(16, cpu_cores - 2)`) will queue any surplus; on a typical 8-core machine, ~6 run concurrently and 4 queue. This is fine — total wall-clock is dominated by the largest single task (B1 at 153 errors).
- **Subagent worktree cleanup**: `Agent({isolation: "worktree"})` auto-removes the worktree if the subagent makes no changes. After a successful diff capture, the worktree path is reported in the result; manually clean up with `git worktree remove <path>` after merging.
