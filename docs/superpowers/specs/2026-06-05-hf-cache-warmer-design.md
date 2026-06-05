# Centralized HF Cache Warmer for CI

**Status:** approved (sections 1-2 explicitly approved by user; user delegated the rest)
**Date:** 2026-06-05
**Owner:** voorhs

## Problem

CI repeatedly fails because of HuggingFace Hub rate limits (1000 requests / 5 minutes per authenticated account):

1. Workflows run in parallel (unit-tests, test-embedder, test-scorers, test-presets, test-optimization, test-inference) — each with up to 6 matrix jobs.
2. On a cold cache, each job downloads the HF models the tests need, in parallel — fast path to quota exhaustion.
3. Once the quota is hit, even cached-files-only operations fail because `transformers` calls `model_info()` from inside tokenizer loads (the `_patch_mistral_regex` bug; mitigated separately by the conftest patch, tracked for full fix in #295).

Symptom families observed in real runs:

- `HfHubHTTPError: 429 Too Many Requests for url: .../api/models/<repo>`
- `huggingface_hub.errors.LocalEntryNotFoundError`
- `OSError: We couldn't connect to 'https://huggingface.co' to load the files`

Per-workflow point fixes (smart prewarm, cache key tweaks, conftest patches) have helped but the root cause remains: every workflow independently tries to warm its corner of the cache.

## Goal

A single CI step that fully populates the HF cache before any test job runs. Once warm, tests find every model and dataset on disk and make zero HF API calls.

Success criteria:

- Cold-cache runs hit HF at most once per model/dataset (in the warm-cache job), not once per workflow × matrix job.
- On steady-state runs the warm-cache job is a near-instant cache hit (no HF API calls at all).
- A single declarative file (`.ci/hf-prewarm.yaml`) is the source of truth for which HF resources tests depend on. Edits to that file invalidate the cache at exactly the right granularity.
- A HuggingFace outage during warm-cache does NOT cancel every PR — tests proceed best-effort.
- Workflow YAML stays thin: control flow / parsing / retry logic lives in a dedicated Python script, not in inline bash.

## Architecture

| Action | Path | Purpose |
|---|---|---|
| **NEW** | `.ci/hf-prewarm.yaml` | Declarative list of HF models + datasets to warm, each pinned to a SHA |
| **NEW** | `.ci/warm_hf_cache.py` | Python script: parse the YAML, prewarm each entry with retries and fast-path |
| **NEW** | `.github/workflows/ci.yaml` | Single orchestrator workflow: `warm-cache` job → all test jobs (`needs: warm-cache`) |
| **MOD** | `.github/workflows/reusable-test.yaml` | Restore-keys prefer the warmed cache; inline prewarm step removed |
| **DEL** | `.github/workflows/unit-tests.yaml` | Folded into `ci.yaml` |
| **DEL** | `.github/workflows/test-embedder.yaml` | Folded into `ci.yaml` |
| **DEL** | `.github/workflows/test-scorers.yaml` | Folded into `ci.yaml` |
| **DEL** | `.github/workflows/test-presets.yaml` | Folded into `ci.yaml` |
| **DEL** | `.github/workflows/test-optimization.yaml` | Folded into `ci.yaml` |
| **DEL** | `.github/workflows/test-inference.yaml` | Folded into `ci.yaml` |

PR status check names change from e.g. `unit tests / test (...)` to `CI / unit-tests / test (...)`. **Branch protection rules will need to be updated** to point at the new check names; this is an out-of-band step that requires repo admin.

## Components

### `.ci/hf-prewarm.yaml`

Flat YAML, two lists:

```yaml
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

**Format:** `<repo_id>@<sha>`. Revisions are mandatory — pinning to a SHA is what lets the fast path skip the HF API entirely (a 40-char hex revision is locally verifiable). Adding an entry with an unpinned revision is a config error caught by the warmer script.

**Single source of truth:** the warmer script reads this; future tooling (a local dev `make warm-cache`, a lint check) can read it too.

### `.ci/warm_hf_cache.py`

Stdlib + `huggingface_hub` + `PyYAML`. Behavior:

1. Loads `.ci/hf-prewarm.yaml`.
2. Validates every entry has the `repo@sha` shape (raises early if not).
3. For each entry, in series (parallelism is what causes 429s):
   - **Fast path:** `snapshot_download(repo_id, revision, repo_type, local_files_only=True)`. If it returns a path, the cache already has every file for that exact revision — log `cached`, continue. **No HF API call.**
   - **Cache miss:** `snapshot_download(...)` with 4 attempts, exponential backoff at 60s / 120s / 180s. Catches `HfHubHTTPError`, `OSError`, `LocalEntryNotFoundError`. Logs `downloaded` on success, `failed` after the last attempt.
4. Prints a one-line summary: `N cached, M downloaded, K failed`.
5. Always exits 0 (best-effort): a failed download is a warning, not a CI failure. The downstream job's cache restore will detect missing files at test time and surface the actual failure with full context.

CLI:
- `uv run python .ci/warm_hf_cache.py` — uses default path `.ci/hf-prewarm.yaml`
- `uv run python .ci/warm_hf_cache.py --config <path>` — for tests / local dev
- `--strict` flag: exit non-zero if anything failed. Off by default for CI; intended for local sanity-check runs.

Runnable locally for dev-machine bootstrap: `cd repo && HF_TOKEN=... uv run --with 'huggingface_hub[hf_xet]' --with pyyaml python .ci/warm_hf_cache.py`.

### `.github/workflows/ci.yaml`

Single orchestrator. Triggers: `push` to `dev`, `pull_request`. Concurrency group cancels stale runs on PR pushes.

Job graph:

```
warm-cache (Linux, Windows)  ◄── continue-on-error
   │
   ├── unit-tests
   ├── test-embedder
   ├── test-scorers (matrix: base, transformers, peft, catboost)
   ├── test-presets
   ├── test-optimization
   └── test-inference
```

Each test job has `needs: warm-cache` and `uses: ./.github/workflows/reusable-test.yaml` with the same `test_command` / `extras` inputs that the previous standalone files carried. No `prewarm_models` input — that goes away.

The warm-cache step is a single line:
```yaml
- name: Run warm-cache script
  env: { HF_TOKEN: '${{ secrets.HF_TOKEN }}' }
  run: uv run --with 'huggingface_hub[hf_xet]' --with pyyaml python .ci/warm_hf_cache.py
```

No inline bash control flow. Everything else (config loading, retries, error handling) lives in Python.

### `.github/workflows/reusable-test.yaml` modifications

1. **Restore-keys** gain two higher-priority entries:
   ```yaml
   restore-keys: |
     ${{ runner.os }}-hf-warmed-${{ hashFiles('.ci/hf-prewarm.yaml') }}-
     ${{ runner.os }}-hf-warmed-
     ${{ runner.os }}-hf-${{ github.run_id }}-
     ${{ runner.os }}-hf-
   ```
   Order matters: try the exact-hash warmed cache first, then any warmed cache (covers cases where someone edited the yaml mid-PR), then the previous fallback chain.
2. The inline bash **prewarm step is removed**. The `prewarm_models` input is removed too. This is a breaking change to the reusable workflow's surface, but the only callers (the 6 test-*.yaml files) are being deleted in the same change.
3. Everything else (cache step, install, run tests) stays as-is.

## Data Flow

1. PR opened or commit pushed to `dev`.
2. `ci.yaml` starts. The `warm-cache` matrix dispatches (Linux + Windows).
3. Each warm-cache job runs `actions/cache@v4` restore:
   - Cache key for save: `${{ runner.os }}-hf-warmed-${{ hashFiles('.ci/hf-prewarm.yaml') }}-${{ github.run_id }}`.
   - Restore-keys cascade prefers the exact-hash warmed cache, then any warmed cache, then any HF cache.
   - On a freshly-edited `hf-prewarm.yaml`, the exact-hash key misses but the cascade catches the previous warmed cache → most files are already on disk, the script only downloads the diff.
4. The warm-cache job runs `warm_hf_cache.py`. Fast-path most entries (`local_files_only=True` succeeds), download anything missing.
5. Job completes; `actions/cache@v4` saves the cache under the exact-hash key.
6. All test jobs (queued via `needs: warm-cache`) start.
7. Each test job's cache restore picks up the same warmed cache via the matching restore-key.
8. Tests run with every model and dataset already on disk. Zero HF API calls outside the warm-cache job.

On subsequent runs of the same `hf-prewarm.yaml`:

- warm-cache restore hits the exact-hash key, fast-path returns `cached` for every entry, total step time ≈ HF cli installation overhead (a few seconds). Zero HF API calls.

## Error Handling

| Scenario | Behavior |
|---|---|
| HF returns 429 on a specific download | Script retries with 60s / 120s / 180s backoff (sized for the 5-min rate window). After 4 attempts, logs `failed`, continues to next entry. |
| HF is completely down for the warm-cache run | Every entry fails after retries. Script exits 0. `continue-on-error: true` lets test jobs run; they fall back to whatever's in the restored cache. The conftest mistral patch and the `DEFAULT_REVISIONS` SHA pins in `src/autointent/configs/_transformers.py` limit damage by preventing extra `model_info` calls during tokenizer loads. |
| Someone adds a new model to `hf-prewarm.yaml` without bumping the SHA file properly | Script validates `@sha` shape; raises before any download, fails the warm-cache step. `continue-on-error` still lets tests run, but the failure is visible in the run summary. |
| `hf-prewarm.yaml` is edited but old warmed cache lingers | Cache key hash changes, exact match misses, but the `${{ runner.os }}-hf-warmed-` restore-key cascade catches the previous warmed cache. Script's fast path skips unchanged entries; only the diff is downloaded. |
| GitHub evicts the warmed cache (after 7d unused) | Cold start: full download once, ~5-10 minutes. Recovers automatically after the next push. |
| Branch protection still points at old check names | PR shows missing checks until rules are updated. Out-of-band repo-admin step, called out in the PR description. |
| `transformers` upgrade lands (issue #295) | Conftest patch removed, warm-cache logic unchanged. No interaction. |

## Testing

**Local sanity check** (the dev-team-prefers-python win):
```bash
HF_TOKEN=... uv run --with 'huggingface_hub[hf_xet]' --with pyyaml python .ci/warm_hf_cache.py --strict
```
Should succeed on a workstation with internet and a valid token. `--strict` is a non-default flag that makes the script exit non-zero on any failed download, useful for the dev loop and CI-of-CI but not for production CI.

**CI smoke runs:**

1. **First push after merge:** GitHub has no `Linux-hf-warmed-*` cache yet. Cascade falls through to `Linux-hf-` (existing wildcard from current branch). The warmer script fast-paths the entries that happen to already be cached and downloads the rest. Total warm-cache step: 5-10 min on Linux, similar on Windows. All test jobs that follow should pass with cache hits.
2. **Second push, same `hf-prewarm.yaml`:** Cache exact-hash hit. warm-cache step ≈ 30 seconds (uv setup + Python startup + 11 fast-path checks). All test jobs cache-hit. Zero HF API calls anywhere.
3. **Edit `hf-prewarm.yaml`** (e.g. add a model): Exact-hash misses, prefix-hash restore-key catches the previous warmed cache, only the new entry is downloaded. Subsequent runs reach steady state.

**What we're NOT testing:**

- Stress-testing the HF backoff (would require sustained quota exhaustion in CI, which is what we're trying to avoid).
- Behavior under network partition. Covered by `continue-on-error`; we trust the cache fallback chain.

## Out of scope

- Upgrading transformers to 5.x (tracked separately in #295). The conftest mistral patch stays as-is — `warm_hf_cache.py` only calls `snapshot_download`, which doesn't instantiate tokenizers and therefore doesn't trigger `_patch_mistral_regex`. The patch is still load-bearing for test runtime where tokenizers actually get loaded.
- Re-architecting how autointent itself loads models.
- Bundling test datasets into git LFS (considered, rejected: user wants CI-level caching).
- Cache warmup on a schedule (cron): not needed once warm-cache runs on every push; GitHub's 7-day eviction is bounded by the natural push cadence on `dev`.

## Risks

1. **Branch protection update is out-of-band.** Whoever merges this needs repo-admin to update branch protection rules pointing at the new `CI / *` check names. Document this in the PR.
2. **First run after merge is slow.** ~5-10 min warm-cache step on cold cache. Acceptable one-time cost.
3. **Cache size.** Rough estimate: e5-large-instruct (2.2GB) + deberta-v3-large (1.5GB) + others (~1.5GB) = ~5 GB. GitHub's per-repo limit is 10GB, so this fits with headroom. If we approach the limit later, drop deberta-v3-large from prewarm (only used by transformers-heavy preset).
4. **SHA staleness.** If we don't update SHAs in `hf-prewarm.yaml` as upstream models change, tests use older revisions than upstream. This is **intentional** — pinning is the whole point of avoiding `model_info` calls. The list should be reviewed when revisions in `DEFAULT_REVISIONS` are bumped.
