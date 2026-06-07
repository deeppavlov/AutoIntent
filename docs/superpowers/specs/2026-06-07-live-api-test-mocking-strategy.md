# Live-API test mocking strategy

**Status:** implemented — see PR and `docs/superpowers/plans/2026-06-07-live-api-test-mocking.md`
**Worktree:** `.claude/worktrees/research+test-mocking-strategy` (branched from `dev`)
**Related issue:** [#299 LLMDescriptionScorer.dump/load drops generator_config](https://github.com/deeppavlov/AutoIntent/issues/299) — out of scope; the refactor xfails 5 tests pending the fix.

## 1. Problem

Many AutoIntent tests gate on env vars (`OPENAI_API_KEY`, `OPENAI_MODEL_NAME`, OpenSearch host, vLLM GPU presence). The CI workflows in `.github/workflows/` **never inject those secrets**. Every gated test currently fires `SKIPPED` and the suite shows green — there is zero per-PR coverage of OpenAI, OpenSearch, or LLM-based scorer/preset code paths.

Estimated silent-skip surface: **40–50 effective test cases per PR run.**

## 2. Decision: full offline CI

**No live-API tests will run in CI at all.** Every currently-skipped OpenAI/OpenSearch test is converted to either:

- a unit test using a **mock `Generator`** or **`respx`**-mocked httpx transport, or
- an integration test using **`testcontainers-python`** for OpenSearch, running per-PR.

Live OpenAI/vLLM behaviour is the responsibility of production monitoring, not the test suite. vLLM tests remain as today's `skipif` gate and are run manually pre-release.

### 2.1 Why no live tests for `structured_output/test_basics.py` either

The four tests in that file assert types only (`isinstance(response, str)`, `isinstance(result, Person)`) — nothing about response content. They test our wrapper's request building, response deserialisation, and retry loop. All of that is exercised by a respx mock that returns a canned JSON body. The only thing a live test catches that a mock doesn't is "did OpenAI change their structured-output API?" — that belongs in prod monitoring, not PR CI.

### 2.2 Why no `openmock` and no `--block-network`

- **`openmock`** has near-zero community adoption; testcontainers is the safe pick for OpenSearch.
- **Network blocking** is owned by the parallel HF-cache work (`2026-06-06-hf-*` specs) and is out of scope here.

## 3. Audit summary

Walked `tests/` (~70 files, ~294 functions).

### 3.1 By external service

| Service | Files | Default CI behavior |
| --- | --- | --- |
| OpenAI chat completions | 8 | silent skip (no key) |
| OpenAI embeddings | 2 (+ shared fixture param) | silent skip (no key) |
| OpenSearch | 1 (parametrized) | skipped (no `localhost:9200`) |
| vLLM | 1 (shared fixture param) | skipped (no CUDA in CI) — **stays skipped** |
| HuggingFace Hub | ~30 | runs against warmed cache (covered by separate work) |

### 3.2 Silent-no-op-in-CI list (target of this work)

- `tests/modules/scoring/test_description_llm.py` (2)
- `tests/generation/structured_output/test_basics.py` (4)
- `tests/generation/structured_output/test_retries.py` (4)
- `tests/generation/structured_output/test_caching.py` (1)
- `tests/embedder/test_openai_backend.py` (9, module-level `pytestmark`)
- `tests/embedder/conftest.py` `backend_configs` — every `id="openai"` param cuts across `test_basic.py`, `test_caching.py`, `test_prompts.py`, `test_dump_load.py`, `test_memory.py`, `test_sentence_transformers_backend.py`
- `tests/pipeline/test_inference.py::*description_with_llm` (2 × N)
- `tests/pipeline/test_optimization.py::*description_with_llm` (3 × N)
- `tests/pipeline/test_presets.py::*zero-shot-llm` (1)
- `tests/generation/utterances/test_balancer.py::test_real_balancer` (delete — duplicates mocked variant)
- `tests/context/test_vector_index.py` OpenSearch param (whole class × N)

### 3.3 Existing mocking patterns (extend, don't replace)

The `tests/generation/utterances/` and `tests/generation/intents/` subsystems already use:

- `Mock(spec=Generator)` / `AsyncMock` for `get_chat_completion[_async]`
- `patch("openai.OpenAI")` for constructor + autouse `set_env_vars` monkeypatch
- `AsyncMock` on `client.get_chat_completion_async`

The scoring, structured-output, embedding-backend, and pipeline-LLM-scorer suites do not use this pattern.

### 3.4 Dead coverage

- `tests/generation/utterances/test_evolver.py` — three `@pytest.mark.skip(reason="issues with sentence-transformers dependency")`.
- `tests/generation/utterances/test_balancer.py::test_real_balancer` — duplicates `test_balancer`.

Both deleted in Phase 1.

## 4. Tool picks

| Need | Pick | Why |
| --- | --- | --- |
| OpenAI client paths (chat + structured output + embeddings) | **`respx`** | Both OpenAI and Anthropic SDKs are httpx wrappers. Anthropic's own SDK uses respx as a dev dep — strongest community signal. |
| Pipeline-level tests where the LLM is incidental | **`Mock(spec=Generator)`** | Already in use; lighter-weight than respx for tests that don't care about the wire. |
| OpenSearch | **`testcontainers-python`** `OpenSearchContainer` | Real container per session; mocking the REST layer just retests our wrapper. |
| vLLM | **leave as-is, skip in CI, manual pre-release** | Requires GPU; out of scope for this work. |

### 4.1 Tools we explicitly rejected

- **VCR.py / pytest-recording** — vcrpy has open async-httpx streaming bugs (#597, #895, #927); cassette maintenance overhead not worth it given respx is sufficient.
- **`openai-responses-python`** — maintainer-flagged "maintenance mode"; raw `respx` is the safer dependency.
- **`openmock`** — near-zero community adoption.
- **`litellm` mock_response** — AutoIntent doesn't route through litellm.

## 5. Verdicts per test family

| Family | Verdict | Mechanism |
| --- | --- | --- |
| `test_openai_backend.py` (9) | **MOCK** | Import alias: `from tests._fixtures.fake_openai_embedding import FakeOpenaiEmbeddingBackend as OpenaiEmbeddingBackend` |
| `test_description_llm.py` + pipeline `description_with_llm` params (~8) | **MOCK** | Shared `patch_llm_scorer_generator` fixture; dump/load split into a separate xfail test pending #299 |
| `structured_output/test_caching.py` (1) | **respx + cache isolation** | Counter via `route.call_count`; `_isolated_cache` autouse fixture for tmp_path |
| `structured_output/test_retries.py` (4) | **respx** | `side_effect=[bad, bad, good]` for success; constant bad for RetriesExceededError |
| `structured_output/test_basics.py` (4) | **respx** | Type-only assertions; canned JSON body satisfies all four tests |
| `test_balancer.py::test_real_balancer` (1) | **DELETE** | Duplicate of mocked `test_balancer` |
| `test_evolver.py` skipped tests (3) | **DELETE** | Indefinitely skipped |
| `test_vector_index.py` OpenSearch param | **testcontainers** | `opensearch_container` session fixture; `"opensearch_lazy"` placeholder string resolved via `request.getfixturevalue` so Faiss-only runs don't boot Docker |
| `embedder/conftest.py` `backend_configs` `id="openai"` | **MOCK** | Autouse `_autouse_fake_openai_embedding` in `tests/embedder/conftest.py` patches `Embedder._init_backend`'s `OpenaiEmbeddingBackend` symbol |
| `embedder/conftest.py` `backend_configs` `id="vllm"` | **leave skipped** | vLLM stays manual pre-release |

## 6. Shared fixture surface

In `tests/_fixtures/`:

- `mock_generator`, `mock_async_generator`, `patch_llm_scorer_generator` — `mock_generator.py`. The patch monkeypatches `autointent.modules.scoring._description.llm_encoder.Generator` (at the import site, not the source module).
- `FakeOpenaiEmbeddingBackend`, `patch_openai_embedding_backend` — `fake_openai_embedding.py`. The patch monkeypatches `autointent._wrappers.embedder.embedder.OpenaiEmbeddingBackend` (covers both `_init_backend` and `Embedder.load` because both resolve via that module's namespace).
- `respx_openai` — `respx_openai.py`. Sets dummy `OPENAI_API_KEY`/`OPENAI_MODEL_NAME`, clears `OPENAI_BASE_URL`, yields a `respx.MockRouter`.
- `opensearch_container` — `opensearch_container.py`. Session-scoped fixture booting an `OpenSearchContainer` and yielding `(host, port)`.

All four re-exported from `tests/conftest.py` at the bottom.

## 7. CI changes

- **No new secrets in any workflow.** No new scheduled workflows.
- **Two new test-group dev deps**: `respx (>=0.21.0,<1.0.0)`, `testcontainers[opensearch] (>=4.5.0,<5.0.0)`.
- **Docker on CI**: GitHub Actions hosted runners already have Docker; no runner change needed.

## 8. Phasing (as implemented)

Each phase landed as independently mergeable commits:

1. **Cleanup.** Delete the 3 evolver skips + `test_real_balancer`.
2. **`mock_generator` + scoring/pipeline LLM tests.** Adds the fixtures + converts `test_description_llm.py` + pipeline `description_with_llm` params. *Bonus*: prod fix for `_init_event_loop` on Python 3.10+ (`asyncio.get_event_loop()` raises when no loop exists). Bonus: 5 tests xfailed pending #299.
3. **`FakeOpenaiEmbeddingBackend` + `embedder/conftest.py`.** Replace `id="openai"` skipif with autouse patch.
4. **`respx` for `test_openai_backend.py` and `structured_output/`.** ~18 tests start running per PR.
5. **`OpenSearchContainer` for `test_vector_index.py`.** OpenSearch tests run per PR via Docker.

vLLM is intentionally out of scope.

## 9. Risks (resolved or noted)

- **Docker startup time on CI** for the OpenSearch testcontainer (~10–15s session cost). Acceptable for the coverage win. Lazy `request.getfixturevalue("opensearch_container")` ensures Faiss-only runs don't boot Docker.
- **#299 prod bug** — the dump/load round-trip in `LLMDescriptionScorer` silently drops `generator_config`. Five tests xfailed strict pending the fix. **Must be fixed and xfails flipped before this branch merges.**

## 10. Out of scope

- HF cache / SHA-pinning (covered by `2026-06-06-hf-*` and `2026-06-07-hf-sha-single-source-design.md`).
- Test-quality concerns unrelated to live-API gating.
- Performance / parallelism of the suite.
- vLLM coverage in CI.
- Network blocking / `pytest-socket` (owned by HF work).
