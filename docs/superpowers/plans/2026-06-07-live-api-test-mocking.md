# Live-API Test Mocking — Implementation log

**Status:** all phases complete. See PR.
**Spec:** `docs/superpowers/specs/2026-06-07-live-api-test-mocking-strategy.md`

Five phases, each independently reviewable.

## Phase 1 — Cleanup (dead code)

- Delete 3 indefinitely-skipped tests in `tests/generation/utterances/test_evolver.py` (`@pytest.mark.skip(reason="issues with sentence-transformers dependency")`).
- Delete `tests/generation/utterances/test_balancer.py::test_real_balancer` (duplicates mocked `test_balancer`).

Commits: `4c2dc65c`, `ce7b384d`.

## Phase 2a — `mock_generator` fixtures + conftest wiring

Create `tests/_fixtures/` package and `mock_generator.py` containing:

- `_make_categorization(most_probable_index=0)` returning `IntentCategorization(reasoning="mocked", most_probable=[index+1], promising=[])`.
- `mock_generator` — `Mock(spec=Generator)` whose `get_structured_output_sync` and `get_chat_completion` return canned values.
- `mock_async_generator` — async variant.
- `patch_llm_scorer_generator(monkeypatch)` — monkeypatches `autointent.modules.scoring._description.llm_encoder.Generator` so `LLMDescriptionScorer` constructs a combined sync+async `Mock(spec=Generator)`.

Re-export the three fixtures from `tests/conftest.py`.

Commits: `9aba49b7`, `b50daca2`, `8d4fbcd1`, `45e764c3` (lint), `684de56a` (drop dead params).

## Phase 2b — `tests/modules/scoring/test_description_llm.py`

Rewrite the file:
- Drop the `OPENAI_API_KEY` skipif.
- Both retained tests take `patch_llm_scorer_generator` as a parameter.
- Split the dump/load round-trip into its own `test_description_scorer_llm_dump_load_roundtrip` decorated `@pytest.mark.xfail(strict=True, reason="… https://github.com/deeppavlov/AutoIntent/issues/299. Flip when fixed.")`. The original test had a dump/load block inline; that block hit a pre-existing prod bug (`LLMDescriptionScorer.dump/load` drops `generator_config`) once the skipif was removed.

Side-quest: fix `_init_event_loop` in `src/autointent/modules/scoring/_description/llm_encoder.py` to handle Python 3.10+'s `asyncio.get_event_loop()` raising when no loop exists (try/except RuntimeError → `asyncio.new_event_loop()`).

Commits: `7f7993d7` (asyncio fix), `448b34e2` (fixture **kwargs + ruff per-file-ignores broadening), `3473361b` (test conversion).

## Phase 2c — Pipeline LLM tests

Convert three files:
- `tests/pipeline/test_optimization.py` — three parametrize blocks; strip skipif on `description_with_llm`, add `patch_llm_scorer_generator` to test signatures.
- `tests/pipeline/test_inference.py` — two parametrize blocks; same treatment.
- `tests/pipeline/test_presets.py` — in-body `pytest.skip()` replaced with fixture injection.

Then add `pytest.mark.xfail(strict=True, reason="… #299")` to the four rows that trip the dump/load bug:
- `test_optimization.py::test_dump_modules[description_with_llm]`
- `test_inference.py::test_inference_from_config[description_with_llm]`
- `test_inference.py::test_inference_on_the_fly[description_with_llm]`
- `test_presets.py::test_presets[zero-shot-llm]`

`test_cv` and `test_no_context_optimization` description_with_llm rows PASS (no dump/load chain).

Commits: `c02f948c`, `fc17c92b`, `72ea9992`, `6ed8827e`.

## Merge from `origin/dev`

Pulled mid-stream after `dev` got the HF rate-limit + warm-cache PRs (#294). Only one content conflict in `tests/conftest.py` (canonical test models vs. fixture re-exports — both retained).

Commit: `ec3ff3ec`.

## Phase 3a — `FakeOpenaiEmbeddingBackend`

Create `tests/_fixtures/fake_openai_embedding.py`:
- `_seeded_vector(text, dim, *, seed_extra="")` — deterministic unit vector using sha256 of `f"{seed_extra}|{text}"` plus per-token additive mixing with weight 0.5.
- `FakeOpenaiEmbeddingBackend(BaseEmbeddingBackend)` implementing the full abstract interface (`__init__`, `clear_ram`, `embed` with overloads, `similarity`, `get_hash`, `dump`, `load`). `dim = config.dimensions or 1536`. `embed()` touches `self._client` on first call to make `test_client_lazy_loading` observable.
- `patch_openai_embedding_backend(monkeypatch)` — one-line `monkeypatch.setattr(embedder_module, "OpenaiEmbeddingBackend", FakeOpenaiEmbeddingBackend)`. Covers both `Embedder._init_backend` (`embedder.py:66`) and `Embedder.load` (`embedder.py:163`).

Re-export from `tests/conftest.py`.

Commit: `fd425acd`.

## Phase 3b — `tests/embedder/conftest.py` swap

- Drop the `openai_available = os.getenv(...)` line.
- Strip `marks=pytest.mark.skipif(...)` from the `id="openai"` parametrize row in `backend_configs`.
- Drop unused `import os`.
- Append an autouse fixture `_autouse_fake_openai_embedding(patch_openai_embedding_backend)` so every test under `tests/embedder/` gets the fake.
- `id="vllm"` parametrize stays gated (vLLM out of scope).

Commit: `0ad39dd6`.

## Phase 4a — `respx` infra

Add to `pyproject.toml` test group: `respx (>=0.21.0,<1.0.0)`.

Create `tests/_fixtures/respx_openai.py`:
- `respx_openai(monkeypatch)` fixture: sets dummy `OPENAI_API_KEY=test-key-not-real` and `OPENAI_MODEL_NAME=gpt-test`, clears `OPENAI_BASE_URL`, yields a `respx.MockRouter(base_url="https://api.openai.com", assert_all_called=False)`.

Re-export from `tests/conftest.py`.

Commits: `54b58522`, `be33f228`.

## Phase 4b — `tests/embedder/test_openai_backend.py`

Rewrite the file header to alias the fake under the existing test name:

```python
from tests._fixtures.fake_openai_embedding import FakeOpenaiEmbeddingBackend as OpenaiEmbeddingBackend
```

Drop the module-level `pytestmark = pytest.mark.skipif(...)` and `import os`. Leave the rest of the class body untouched.

The 10 tests now verify the fake's shape contract (not OpenAI's behaviour). Per spec §2.1.

Commit: `79e56f2c`.

## Phase 4c — `tests/generation/structured_output/`

Three files converted:
- `test_basics.py` — 4 tests with type-only assertions on `Person`. Each test registers a route on `respx_openai`.
- `test_retries.py` — 4 tests covering success-after-retries (`side_effect=[invalid, invalid, valid]`) and `RetriesExceededError` (constant invalid). `Person` keeps its `model_validator`; `VALID_PERSON_JSON` satisfies it.
- `test_caching.py` — 1 test asserting cache via `route.call_count == 2`. Adds `_isolated_cache` autouse fixture that monkeypatches `autointent.generation._cache.user_cache_dir` to return `tmp_path` (otherwise the test writes to the real user cache dir).

Approved deviation from initial draft: the `generator` fixtures take `respx_openai` as a parameter so the env vars are set BEFORE `Generator()` constructs (Generator reads `OPENAI_API_KEY` at init).

Commits: `22e9ba5b`, `4684c99e`, `bed42f2e`.

## Phase 5a — testcontainers OpenSearch infra

Add to `pyproject.toml` test group: `testcontainers[opensearch] (>=4.5.0,<5.0.0)`.

Create `tests/_fixtures/opensearch_container.py`:
- Session-scoped `opensearch_container` fixture. Lazy-imports `OpenSearchContainer` inside the body so users without Docker can still collect tests. Yields `(host, port)`.

Re-export from `tests/conftest.py`.

Commits: `25341330`, `92767a97`.

## Phase 5b — `tests/context/test_vector_index.py`

- Drop the `opensearch_available` module-level block and the `is_opensearch_running()` function.
- Replace `backend_configs` so the OpenSearch row is `pytest.param("opensearch_lazy", id="opensearch")` (no skipif).
- The `vector_index` fixture takes `request` (NOT `opensearch_container` directly) and calls `request.getfixturevalue("opensearch_container")` only inside the `if vector_config == "opensearch_lazy":` branch. This keeps Faiss-only runs Docker-free.
- `test_initialization` branches on `isinstance(vector_config, str)` for the placeholder case.
- `test_opensearch_dependency_error` (which mocks `__import__`) unchanged.

Commits: `1cdb136a`, `0dd6da91` (lazy container fix).

## Final whole-branch review

Passed. One follow-up suggested: clean up the local `mock_generator` fixture in `tests/generation/utterances/test_balancer.py` that shadows the new shared fixture (intentional in dev, now redundant). Non-blocking.

## Blocker on this PR

[Issue #299](https://github.com/deeppavlov/AutoIntent/issues/299) must be fixed and the 5 xfail markers flipped to pass before this PR merges:

- `test_description_llm.py::test_description_scorer_llm_dump_load_roundtrip[True]`
- `test_description_llm.py::test_description_scorer_llm_dump_load_roundtrip[False]`
- `test_inference.py::test_inference_from_config[description_with_llm]`
- `test_inference.py::test_inference_on_the_fly[description_with_llm]`
- `test_optimization.py::test_dump_modules[description_with_llm]`
- `test_presets.py::test_presets[zero-shot-llm]`

(That's 4 xfail markers covering 6 logical test runs after the `[True]`/`[False]` parametrize.)
