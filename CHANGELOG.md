# Changelog

All notable changes to this project are documented in this file. Release notes are grouped by theme rather than listing every commit.

## [0.3.2] — 2026-06-22

Compared to [0.3.1](https://github.com/deeppavlov/AutoIntent/releases/tag/v0.3.1). A maintenance release focused on caching correctness and CI/test coverage. No breaking changes.

### Bug fixes

- **Structured-output cache** now keys entries by **model identity** — the model name and API `base_url` are folded into the cache key alongside the messages, schema, and generation params, so switching models or endpoints no longer returns another model's cached completion (#336).
- **Structured-output cache** eager loading and eviction now treat each on-disk entry as a **directory** (the layout `PydanticModelDumper` actually writes). The previous `is_file()` filter matched nothing and silently disabled eager cache loading; eviction now uses `rmtree` instead of `unlink` (#331).
- **Offline embedding cache key** is now stable: when `HF_HUB_OFFLINE` is set, the commit SHA is read from the local Hugging Face ref file (`$HF_HUB_CACHE/<repo>/refs/<rev>`) so the key matches the online path instead of falling back to the bare revision string. The embedding cache key now also incorporates the **model name**, preventing cross-model cache collisions for non-local models (#337).
- **Augmentation console scripts** `basic-aug` and `evolution-aug` now resolve to the correct module paths (`autointent.generation.utterances._basic.cli` / `._evolution.cli`); the entry points were left dangling by an earlier module rename (#330).

### Tooling, CI, and tests

- **Coverage regression floor**: the combined coverage total is now gated against an **85%** floor, so a drop below it fails CI (#333).
- **Manual coverage dispatch** workflow added and the broken coverage configuration fixed (#325).
- **Windows test jobs** restored — a bash-only shell step had been breaking them on Windows runners (#328).
- Added test coverage for the **`Generator`** `dump`/`load` round-trip and the async empty-response guard (#332), the **basic/evolution augmentation CLIs** (#330), and the **`JSONFormatter`** plus **macro retrieval metrics** (#329).

---

## [0.3.1] — 2026-06-16

Compared to [0.3.0](https://github.com/deeppavlov/AutoIntent/releases/tag/v0.3.0). A maintenance release focused on bug fixes, reproducibility of default Hugging Face downloads, and CI/test stability. No breaking changes.

### Bug fixes

- **LLM description scorer** now persists `generator_config` across `dump`/`load`, so a reloaded `LLMDescriptionScorer` rebuilds its `Generator` with the original settings instead of defaults (#302).
- **Tunable decision threshold** test fixture updated for **Optuna 4.9** sampling order — confirms compatibility with `optuna>=4.9` (#293).
- **faiss** typing tweak for `IndexFlatIP` to satisfy stricter `mypy` runs (#292).

### Reproducible Hugging Face defaults

- New canonical SHA pin map `autointent.configs._pinned_revisions.DEFAULT_REVISIONS`, applied automatically by `HFModelConfig` when `revision` is left unset. Default embedders, rerankers, and NLI models now resolve to fixed commit SHAs instead of `main`, making installs deterministic and avoiding Hub `429` rate limits under parallel CI load. To opt out, pass an explicit `revision` (including `"main"`) on the relevant config (#294).

### Public API additions

- New `RetrievalMetricFnWithOOS` and `ScoringMetricFnWithOOS` protocols (exported from `autointent.metrics`) describing the OOS-aware metric callables returned by the existing `ignore_oos` decorators (#316).

### Tooling, CI, and tests

- **Full OS × Python matrix** is now gated behind the `full-ci` PR label and `dev` pushes; ordinary PRs run a faster default matrix (#318).
- **`mypy --strict`** is enforced on `tests/` to prevent type drift between fixtures and production code (#316).
- **Soft `mypy` profile** added for `docs/` and `user_guides/` so tutorials and Sphinx helpers are type-checked without forcing them onto strict mode (#320).
- **Live-API tests** (OpenAI, OpenSearch) are mocked in CI — no network calls or credentials required to run the suite locally (#301).
- **HF cache prewarm** workflow plus a `.ci/warm_hf_cache.py` helper keyed off `DEFAULT_REVISIONS` removes the long-standing Hub rate-limit flakes in the matrix (#294).
- **GitHub Actions** runners bumped to Node 24 images (#300).
- **Pipeline interruption tests** restored after the sampler API removal in #296 (#303).
- Test layout cleanup: the legacy `setup_environment` helper is gone — tests now use the standard pytest `tmp_path` fixture (#317).

---

## [0.3.0] — 2026-05-19

Compared to [0.2.0](https://github.com/deeppavlov/AutoIntent/releases/tag/v0.2.0).

### Breaking: slimmer default install and optional extras

Heavy integrations are now **optional** and loaded **lazily** where possible. A plain `pip install autointent` pulls a smaller core set of dependencies; features that need transformers, OpenSearch, MCP, OpenAI, vLLM, and similar stacks require the matching **optional dependency groups** (see `pyproject.toml`). If you relied on everything being importable after a minimal install, pin extras explicitly in your environment.

### Embeddings: OpenAI, vLLM, and training

- **OpenAI embeddings** path with safer batching and tokenizer fallbacks when the API does not expose `tokenizer` metadata.
- **vLLM** optional extra for serving-compatible embedding workflows.
- **Embedder fine-tuning** utilities and training-oriented improvements for custom embedding models.

### Scoring and search

- **GCN-based scorer** for graph-style intent scoring in the AutoML search.
- **OpenSearch** optional backend for vector index storage and retrieval alongside existing options.

### Serving and integration

- **MCP (Model Context Protocol)** interface plus **HTTP** serving pieces (FastAPI / related extras) for running AutoIntent behind an API or MCP host.

### Data, splits, and augmentation

- **Out-of-scope (OOS) intents** are **always kept separate** in dataset splits so evaluation stays honest.
- **Adversarial augmentation** support for more robust pipelines under optimization.

### Configuration and validation

- **`OptimizationConfig.from_preset`** for constructing optimization settings from named presets.
- **Split readiness checks** (including multilabel-aware validation) so you catch bad label or split geometry before long runs.

### Tooling and dependency bounds

- **uv**-centric developer workflow (dependency groups, reproducible installs).
- **Python** supported through **below 3.15** (`requires-python = ">=3.10,<3.15"` in `pyproject.toml`).
- **OpenAI Python SDK v2** (`openai>=2,<3`) in the `openai` extra.
- **`datasets` pinned below 5** (`datasets>=3.2,<5`) for compatibility with current loaders and tests.

---

## [0.2.0]

See the [v0.2.0 release](https://github.com/deeppavlov/AutoIntent/releases/tag/v0.2.0) on GitHub for earlier changes.
