# Design — Strict mypy on `tests/`

**Date**: 2026-06-07
**Branch**: `b/mypy-on-tests`
**Motivating bug**: [#296](https://github.com/deeppavlov/AutoIntent/issues/296) — a test silently broke against a removed `Pipeline.fit(sampler=...)` kwarg, undetected by CI for weeks.

## Goal

Run `mypy --strict` over `tests/` in CI so test code drifting from typed `src/` APIs fails type-check before it can rot in an orphaned file.

## Scope

In:
- Extend the existing `Typing` CI job to type-check `tests/` alongside `src/autointent`.
- Fix every mypy error in `tests/` so the strict check is green on merge.
- Add the minimum config overrides needed where `src/` already has them.

Out (explicitly excluded — do not bundle):
- Refactors unrelated to type-checking.
- Adding tests, removing tests, restructuring fixtures.
- Source-side type fixes — unless a test reveals a real `src/` signature bug, in which case it is logged and either fixed in a focused commit on this branch or deferred to a separate issue (decision at integration time, see Phase C).
- Loosening the existing `src/` mypy config.

## Baseline

`uv run --group typing mypy tests/` on `dev` reports **745 errors in 78 of 107 files**. Distribution by top-level subdirectory (errors only, excluding mypy notes; sums to 745):

| Subdirectory | Errors |
|---|---|
| `tests/modules` | 249 |
| `tests/embedder` | 108 |
| `tests/data` | 78 |
| `tests/generation` | 65 |
| `tests/configs` | 60 |
| `tests/pipeline` | 41 |
| `tests/conftest.py` | 38 |
| `tests/context` | 24 |
| `tests/server` | 19 (will be excluded via override) |
| `tests/_transformers` | 19 |
| `tests/metrics` | 17 |
| `tests/ci` | 10 |
| `tests/_fixtures` | 10 |
| `tests/callback` | 7 |

`tests/_helpers` is already clean (0 errors). `tests/assets`, `tests/logs`, `tests/__init__.py` contain no Python source picked up by mypy.

Top error codes: `[no-untyped-def]` 413, `[arg-type]` 156, `[no-untyped-call]` 89, `[attr-defined]` 36, `[assignment]` 13, `[union-attr]` 10.

## Strictness profile

Match `src/` exactly — no relaxation for tests. Concretely: the global `[tool.mypy]` block (`strict = true`, `warn_redundant_casts`, `warn_unreachable`, `local_partial_types`, `disable_error_code = ["override"]`, `pydantic.mypy` plugin) applies to `tests/` as-is. The decision driver: writing typed tests is now low-friction with AI assistance, and the value of catching #296-class drift is highest in tests because tests exercise the typed API surface most aggressively.

## Configuration changes

### `.github/workflows/typing.yml`

```diff
-      - name: Run mypy
-        run: uv run mypy src/autointent
+      - name: Run mypy
+        run: uv run mypy src/autointent tests
```

The `typing` group already pulls the extras needed to import the modules `tests/` imports.

### `pyproject.toml`

Add a `tests.server.*` exemption (mirrors `src/autointent.server.*`):

```toml
[[tool.mypy.overrides]]
module = ["tests.server.*"]
ignore_errors = true
```

Rationale: the server modules themselves carry `ignore_errors = true` (pyproject.toml lines 307–312). Strict-typing tests for code that's exempt from strict typing is meaningless.

Extend the existing `ignore_missing_imports` override (lines 275–298) to add modules that test code imports but lack stubs and aren't worth typing on this branch:

```toml
# add to the existing [[tool.mypy.overrides]] module list:
"testcontainers.opensearch",
"warm_hf_cache",  # tests/ci/test_warm_hf_cache.py imports from .ci/ script via sys.path manipulation
```

These are the only two `[import-untyped]` / `[import-not-found]` cases in the baseline. If a subagent surfaces another genuinely-missing-stubs case, it goes here (Phase C decision); subagents do not edit `pyproject.toml` directly.

No new dependencies (pytest ships its own stubs since 6.x; we pin ≥8.3).

## Policy for hard-to-type pytest patterns

1. **Fixtures**: declare return types. pytest-provided fixtures (`tmp_path: Path`, `caplog: LogCaptureFixture`, `monkeypatch: MonkeyPatch`, `capsys: CaptureFixture[str]`) come typed; rely on those signatures. For yield-style fixtures, annotate the return type as `Iterator[T]` (from `collections.abc`) — not `Generator[T, None, None]` unless `send()`/`throw()` semantics are actually used.
2. **`@pytest.mark.parametrize`**: keep `argvalues` as homogeneous tuples or lists; annotate the test fn parameters. If a parametrize set genuinely mixes types, split into multiple parametrize blocks rather than `Any`-ing the parameter.
3. **`unittest.mock.patch` / `MagicMock`**: at the call site, annotate the bound name as `MagicMock`, or `cast(Foo, mock)` when downstream code needs the real type. Do **not** widen test fn signatures to `Any` to accommodate mocks.
4. **Escape hatch**: `# type: ignore[<code>]` (or `# type: ignore[<code1>, <code2>]` when one suppression spans multiple codes) is allowed but must (a) carry specific error codes, never bare, (b) carry an inline `# reason: ...` comment on the same line or the line above. Bare `# type: ignore` is forbidden — `warn_unused_ignores` (implied by strict) will already reject it. Each subagent's diff is reviewed for ignore usage in Phase C.
5. **Pydantic `**kwargs` spread**: `pyproject.toml` sets `init_forbid_extra = true` for the `pydantic.mypy` plugin, which rejects `Model(**kwargs)` as `[call-arg]` even when `kwargs: dict[str, Any]`. Test factories that spread `**overrides` into a model constructor must instead either (a) enumerate the relevant fields explicitly, or (b) build a typed `dict` matching the model's field set and spread that, or (c) at the call site, `cast(<Model>, Model.model_construct(**overrides))` when the test specifically wants to bypass validation. Do not weaken `init_forbid_extra` for tests — that would hide real call-site mistakes.
6. **`pytest.skip` and `warn_unreachable`**: a common pattern is `if not feature_available: pytest.skip(...)` followed by code that mypy considers unreachable after narrowing. `pytest.skip` is typed `NoReturn`, but mypy's narrowing of module-level guards can still mark *test body* code as unreachable in some configurations. When this triggers, prefer `pytest.importorskip("pkg")` (returns the module typed as `ModuleType`) or guard via `pytest.mark.skipif` (decorator) so the test body remains reachable. As a last resort, `# type: ignore[unreachable]` with a reason.
7. **`disable_error_code = ["override"]` is a known blind spot**: the global config disables Liskov-violation warnings, so tests that subclass `BaseModel` / `nn.Module` with mismatched signatures will type-check even when the signatures genuinely diverge. Subagents must not rely on `[override]` to catch base-class signature drift. (This is shared with `src/`; not test-specific. Documented here so reviewers don't assume strict catches it.)

## Execution plan (high-level)

Detailed step-by-step ordering, dependencies, and verification commands belong in the implementation plan (`writing-plans` skill). Below is the architecture only.

### Phase A — Main thread, sequential

1. Worktree off fresh `dev` on branch `b/mypy-on-tests` (done before this spec was written).
2. **Commit 1** — this spec.
3. **Commit 2** — implementation plan (produced by `writing-plans` skill, lives at `docs/superpowers/plans/`).
4. **Commit 3 — Infra**: edit `typing.yml` to add `tests` to the mypy invocation; add the `tests.server.*` override + the two `ignore_missing_imports` additions (`testcontainers.opensearch`, `warm_hf_cache`) to `pyproject.toml`; set the mypy step `continue-on-error: true` so warn-only mode protects the branch during fan-out.
5. **Commit 4 — Shared-surface fixes (sequential, must precede fan-out)**: type the modules that subagents depend on transitively:
   - `tests/conftest.py` (38)
   - `tests/_fixtures/**` (10)
   - `tests/_transformers/**` (19)
   - `tests/_helpers/**` (already 0 errors; locked from modification but no work needed)

   Total: 67 errors. These produce a stable type surface for every other test file. After this commit, `mypy src/autointent tests/conftest.py tests/_fixtures tests/_helpers tests/_transformers` must exit clean.

**Phase B subagent worktrees fork from Commit 4 (the shared-surface fix commit).** No subagent starts before Phase A is fully committed.

### Phase B — Subagent fan-out, parallel

Ten subagents, each in its own git worktree off the Phase A head (specifically, off Commit 4 — the shared-surface-fix commit), each tasked with **zero mypy errors in its assigned subdirectory**.

| # | Worktree branch | Scope | Baseline errors |
|---|---|---|---|
| 1 | `b/mypy-on-tests-modules-scoring` | `tests/modules/scoring` | 153 |
| 2 | `b/mypy-on-tests-modules-decision` | `tests/modules/decision` (incl. `decision/conftest.py`) | 51 |
| 3 | `b/mypy-on-tests-modules-rest` | `tests/modules/{embedding,test_dumper.py,test_regex.py}` | 45 (19+21+5) |
| 4 | `b/mypy-on-tests-embedder` | `tests/embedder` (incl. `embedder/conftest.py`) | 108 |
| 5 | `b/mypy-on-tests-data` | `tests/data` | 78 |
| 6 | `b/mypy-on-tests-generation` | `tests/generation` | 65 |
| 7 | `b/mypy-on-tests-configs` | `tests/configs` | 60 |
| 8 | `b/mypy-on-tests-pipeline` | `tests/pipeline` | 41 |
| 9 | `b/mypy-on-tests-context` | `tests/context` | 24 |
| 10 | `b/mypy-on-tests-misc` | `tests/{callback,ci,metrics}` (incl. `ci/conftest.py`; server excluded via override; assets/logs/`__init__.py` have no checkable code) | 34 (7+10+17) |

**`tests/modules` is pre-split into 3 subagents** because (a) a single 249-error / 32-file subdirectory risks subagent context blowout, (b) the natural `scoring`/`decision`/other split lines up with the module taxonomy, and (c) splitting up-front costs nothing and avoids the failed-subagent-round reactive recovery.

**Per-subdir `conftest.py` ownership**: each subagent owns the `conftest.py` *inside its subdirectory* (e.g., `tests/embedder/conftest.py` belongs to subagent 4, `tests/modules/decision/conftest.py` to subagent 2, `tests/ci/conftest.py` to subagent 10). Only the *root* `tests/conftest.py` is frozen by Phase A. Subagents may re-type the arguments their per-subdir conftests *consume from* the frozen root conftest (e.g., a fixture that takes `dataset: Dataset` as a parameter) — that is annotating, not modifying the frozen surface.

Each subagent contract (full text in the impl plan):
- Read this spec.
- Run `uv run --group typing mypy tests/<dir>` to confirm baseline.
- Fix errors per the policy above.
- Re-run mypy on its dir → expect 0 errors.
- Push the subagent branch to origin and open a PR against `b/mypy-on-tests` (the parent integration branch). GitHub Actions runs the full pytest matrix (`ci.yaml`) and `typing.yml` on the PR — that is the verification surface. **Do not run pytest locally, not even `pytest --collect-only`**: an earlier execution froze the user's host when 10 parallel subagents simultaneously loaded torch / transformers / sentence-transformers via `pytest --collect-only`. The main thread polls `gh pr checks` on each PR before merging.
- Report: diff, mypy exit status, PR URL, list of `# type: ignore` usages added (with codes and reasons).
- **Hard limits**: subagent must not modify `src/`, the root `tests/conftest.py`, `tests/_fixtures/`, `tests/_helpers/`, `tests/_transformers/`, `pyproject.toml`, or any CI file. The subagent's *own* per-subdir `conftest.py` is owned by it (in scope). If a fix needs to touch a frozen path, the subagent records the request in its report and applies a local `cast()` or scoped `# type: ignore[...]` to avoid blocking on it; Phase C decides whether to honor the request.

### Phase C — Main thread, sequential

1. Cherry-pick or merge each subagent's diff into `b/mypy-on-tests`.
2. Run `uv run --group typing mypy src/autointent tests` → expect 0 errors. Resolve any cross-subdir issues (rare; mostly stale ignores after a shared type changed).
3. Run the affected pytest job subsets locally or via a temporary CI push (see `feedback_ci_for_long_tests` memory — defer to CI for long suites).
4. Review every `# type: ignore` added across the diff. Each must have a code and a reason. Reject lazy ignores.
5. If any subagent flagged a real `src/` type bug, decide: fix here (focused commit, no scope creep) or defer to a new issue (`cast()` in the test with a comment linking the issue).
6. **Flip the gate**: remove `continue-on-error: true` from the mypy step. Commit.
7. Push branch, open PR against `dev`. **Do not merge** — leave for user review.

## Risks

- **Subagent silently changes behavior while "fixing types"** (narrows `Any` to wrong concrete type, swallows a failure path, replaces a real call with a mock). Mitigation: each subagent opens a PR against `b/mypy-on-tests`, and `ci.yaml` runs the full pytest matrix on that PR's diff (catches fixture wiring, import-time breakage, and assertion regressions). The main thread blocks the merge into `b/mypy-on-tests` on `gh pr checks` green. Phase C re-runs full mypy and spot-reviews diffs. Any test that flips from failing-with-ignore to passing-with-wrong-type is a code smell to question during review.
- **Cross-subdir conflicts via shared fixtures**. Mitigation: Phase A freezes the shared surface; subagents are prohibited from touching it.
- **A subagent runs out of context fixing its assigned subdir**. Mitigation: `tests/modules` is pre-split into 3 (scoring/decision/rest); the largest remaining single-subagent load is `tests/modules/scoring` at 153 errors across 17 files, which prior bulk-type-fix sessions handle comfortably. If any subagent reports partial completion, Phase C splits the remainder by file count into a follow-up subagent pass.
- **Real src/ type bugs surface**. Mitigation: documented escalation path (Phase C step 5).
- **Subagent local resource exhaustion** (originally framed as "pytest is too slow", but the failure mode that actually fired in practice was OOM — `pytest --collect-only` across 10 parallel subagents loaded torch / transformers / sentence-transformers into memory ten times concurrently and froze the host). Mitigation: subagents do not run pytest at all, not even `--collect-only`. They push their branch and open a PR against `b/mypy-on-tests`; `ci.yaml` runs the full pytest matrix on GitHub Actions on each PR's diff. Local subagent work is mypy-only, which is light enough that 10 parallel processes are not a memory concern.
- **uv sync contention across 10 parallel worktrees**. Mitigation: subagents are launched with `Agent({isolation: "worktree"})`, which serializes worktree creation; uv's global cache (`~/.cache/uv` / `~/Library/Caches/uv`) shares downloaded artifacts across worktrees so each subsequent sync is mostly hardlink work. If contention shows up in practice, Phase B falls back to staggered launches (sets of 5).
- **Pytest types regressing under a future pytest upgrade** (out-of-scope risk, but worth a note). Mitigation: the gate is enforced; an upgrade that breaks types fails CI loudly.

## Rollback

Warn-only mode in Phase A through Phase B keeps `b/mypy-on-tests` from ever being CI-red mid-flight. The final flip is one line in `typing.yml`. If the PR is rejected, deleting the branch reverts everything; no other branch is affected.

## Open questions deferred to the implementation plan

These are real questions the impl plan or Phase A discovery must answer; they are not blockers for the spec.

- **Helper signature conflict at Commit 4.** If a subagent in Phase B needs a *different* annotation for a fixture in the (frozen) `tests/_fixtures/` than what Commit 4 chose, the subagent applies a local `cast()` and records the request in its report. Phase C decides whether to widen the helper's type and re-run mypy across affected subdirs. No subagent re-runs because of this.
- **Pytest plugin coverage**: `pytest-asyncio` and `pytest-rerunfailures` are pinned in deps. The impl plan must spot-check that their public types resolve under the `typing` group; if not, add to `ignore_missing_imports` in Commit 3.
- **Per-extra install in Phase B**: subagents do not run pytest (CI does — see Risks). They do run mypy locally, which means each worktree's venv needs the extras whose stubs / source the test code transitively imports. The impl plan uses a maximal install for every subagent (matches the typing.yml workflow's invocation), trading disk for simplicity and amortized via uv's global cache. If install time becomes a bottleneck, the per-subagent extras mapping lives in `.github/workflows/ci.yaml` and can be cribbed.
- **Mock-patching private `src/` modules**: tests like `tests/_fixtures/mock_generator.py` patch `autointent.modules.scoring._description.llm_encoder`. Strict typing of patched-attribute access may need `cast(Any, ...)` patterns. Policy applies (escape hatch with code + reason); call out in the impl plan that this pattern is *expected* in the frozen `_fixtures/` work in Commit 4.

## Done criteria

1. `uv run --group typing mypy src/autointent tests` exits clean on the final commit.
2. All existing `pytest` CI jobs remain green on the PR.
3. `typing.yml` mypy step is enforced (no `continue-on-error`).
4. No new runtime or dev dependencies.
5. Every `# type: ignore` added has a specific code and an inline reason.
6. PR is open against `dev`, awaiting user review — **not merged**.
