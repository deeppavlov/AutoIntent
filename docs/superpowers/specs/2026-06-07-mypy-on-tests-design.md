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

Add one override to mirror what `src/autointent.server.*` already gets:

```toml
[[tool.mypy.overrides]]
module = ["tests.server.*"]
ignore_errors = true
```

Rationale: the server modules themselves carry `ignore_errors = true` (pyproject.toml lines 307–312). Strict-typing tests for code that's exempt from strict typing is meaningless — the test signatures depend on untyped server symbols.

No other test-side overrides. No new dependencies (pytest ships its own stubs since 6.x; we pin ≥8.3).

## Policy for hard-to-type pytest patterns

1. **Fixtures**: declare return types. pytest-provided fixtures (`tmp_path: Path`, `caplog: LogCaptureFixture`, `monkeypatch: MonkeyPatch`, `capsys: CaptureFixture[str]`) come typed; rely on those signatures.
2. **`@pytest.mark.parametrize`**: keep `argvalues` as homogeneous tuples or lists; annotate the test fn parameters. If a parametrize set genuinely mixes types, split into multiple parametrize blocks rather than `Any`-ing the parameter.
3. **`unittest.mock.patch` / `MagicMock`**: at the call site, annotate the bound name as `MagicMock`, or `cast(Foo, mock)` when downstream code needs the real type. Do **not** widen test fn signatures to `Any` to accommodate mocks.
4. **Escape hatch**: `# type: ignore[<specific-code>]` is allowed but must (a) carry a specific error code, (b) carry an inline `# reason: ...` comment on the same line or the line above. Bare `# type: ignore` is forbidden — `warn_unused_ignores` (implied by strict) will already reject it. Each subagent's diff is reviewed for ignore usage in Phase C.

## Execution plan (high-level)

Detailed step-by-step ordering, dependencies, and verification commands belong in the implementation plan (`writing-plans` skill). Below is the architecture only.

### Phase A — Main thread, sequential

1. Worktree off fresh `dev` on branch `b/mypy-on-tests` (done before this spec was written).
2. Commit this spec.
3. **Infra commit**: edit `typing.yml` to add `tests` to the mypy invocation, add the `tests.server.*` override to `pyproject.toml`, and set the mypy step `continue-on-error: true` so warn-only mode protects the branch during fan-out.
4. **Shared-surface fixes (sequential, must precede fan-out)**: type the modules that subagents depend on transitively:
   - `tests/conftest.py` (38)
   - `tests/_fixtures/**` (10)
   - `tests/_transformers/**` (19)
   - `tests/_helpers/**` (already 0 errors; locked from modification but no work needed)

   Total: 67 errors. These produce a stable type surface for every other test file. After this commit, `mypy src/autointent tests/conftest.py tests/_fixtures tests/_helpers tests/_transformers` must exit clean.

### Phase B — Subagent fan-out, parallel

Eight subagents, each in its own git worktree off the Phase A head, each tasked with **zero mypy errors in its assigned subdirectory**.

| # | Worktree branch | Scope | Baseline errors |
|---|---|---|---|
| 1 | `b/mypy-on-tests-modules` | `tests/modules` | 249 |
| 2 | `b/mypy-on-tests-embedder` | `tests/embedder` | 108 |
| 3 | `b/mypy-on-tests-data` | `tests/data` | 78 |
| 4 | `b/mypy-on-tests-generation` | `tests/generation` | 65 |
| 5 | `b/mypy-on-tests-configs` | `tests/configs` | 60 |
| 6 | `b/mypy-on-tests-pipeline` | `tests/pipeline` | 41 |
| 7 | `b/mypy-on-tests-context` | `tests/context` | 24 |
| 8 | `b/mypy-on-tests-misc` | `tests/{callback,ci,metrics}` (server excluded via override; assets/logs/`__init__.py` have no checkable code) | 34 (7+10+17) |

Each subagent contract (full text in the impl plan):
- Read this spec.
- Run `uv run --group typing mypy tests/<dir>` to confirm baseline.
- Fix errors per the policy above.
- Re-run mypy on its dir → expect 0 errors.
- Run `uv run pytest tests/<dir>` (with appropriate `--extra` flags) to confirm no behavioral regression.
- Report: diff, mypy exit status, pytest exit status, list of `# type: ignore` usages added (with codes and reasons).
- **Hard limits**: subagent must not modify `src/`, `tests/conftest.py`, `tests/_fixtures/`, `tests/_helpers/`, `tests/_transformers/`, `pyproject.toml`, or any CI file. If a fix needs one of these, the subagent records the request in its report and leaves the test untouched.

### Phase C — Main thread, sequential

1. Cherry-pick or merge each subagent's diff into `b/mypy-on-tests`.
2. Run `uv run --group typing mypy src/autointent tests` → expect 0 errors. Resolve any cross-subdir issues (rare; mostly stale ignores after a shared type changed).
3. Run the affected pytest job subsets locally or via a temporary CI push (see `feedback_ci_for_long_tests` memory — defer to CI for long suites).
4. Review every `# type: ignore` added across the diff. Each must have a code and a reason. Reject lazy ignores.
5. If any subagent flagged a real `src/` type bug, decide: fix here (focused commit, no scope creep) or defer to a new issue (`cast()` in the test with a comment linking the issue).
6. **Flip the gate**: remove `continue-on-error: true` from the mypy step. Commit.
7. Push branch, open PR against `dev`. **Do not merge** — leave for user review.

## Risks

- **Subagent silently changes behavior while "fixing types"** (narrows `Any` to wrong concrete type, swallows a failure path, replaces a real call with a mock). Mitigation: each subagent runs `pytest tests/<dir>` post-fix and reports the exit; Phase C reviews diffs and any test that flips from failing-with-ignore to passing-with-wrong-type is a code smell to question.
- **Cross-subdir conflicts via shared fixtures**. Mitigation: Phase A freezes the shared surface; subagents are prohibited from touching it.
- **A subagent runs out of context fixing 274 errors in `tests/modules`**. Mitigation: if the modules subagent reports partial completion, Phase C splits the remainder by sub-subdirectory (`tests/modules/scoring`, `tests/modules/decision`, etc.) into a follow-up subagent pass.
- **Real src/ type bugs surface**. Mitigation: documented escalation path (Phase C step 5).
- **Pytest types regressing under a future pytest upgrade** (out-of-scope risk, but worth a note). Mitigation: the gate is enforced; an upgrade that breaks types fails CI loudly.

## Rollback

Warn-only mode in Phase A through Phase B keeps `b/mypy-on-tests` from ever being CI-red mid-flight. The final flip is one line in `typing.yml`. If the PR is rejected, deleting the branch reverts everything; no other branch is affected.

## Done criteria

1. `uv run --group typing mypy src/autointent tests` exits clean on the final commit.
2. All existing `pytest` CI jobs remain green on the PR.
3. `typing.yml` mypy step is enforced (no `continue-on-error`).
4. No new runtime or dev dependencies.
5. Every `# type: ignore` added has a specific code and an inline reason.
6. PR is open against `dev`, awaiting user review — **not merged**.
