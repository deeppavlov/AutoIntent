# Changelog

All notable changes to this project are documented in this file. Release notes are grouped by theme rather than listing every commit.

## [0.3.3.dev0] — 2026-08-13

Compared to [0.3.2](https://github.com/deeppavlov/AutoIntent/releases/tag/v0.3.2). A development pre-release focused on OpenSearch vector-index correctness and a stricter optional-dependency guard.

### Vector index (OpenSearch)

- **Dumps are now self-contained snapshots.** `OpenSearchBackend.dump()` copies the live index server-side into a write-blocked, immutable generation index (`{base}-best-{uuid}`) recorded in `remote_manifest.json`, instead of writing a bare reference to the live index that every subsequent HPO trial rewrites. `load()` binds read-only to the verified generation and fails loudly when it is missing, was recreated, or the manifest is malformed; manifest-less dumps written by earlier versions keep the old reference semantics. Data never leaves the cluster. New public `remove_module_dump()` deletes a dump tree together with the cluster indices its manifests reference, and the HPO best-trial replacement now goes through it (#346, closes #343).
- **`clear_ram()` no longer destroys durable state.** The `BaseIndexBackend` contract is split: `clear_ram()` releases local resources only, and a new `reset()` drops all indexed documents. `OpenSearchBackend.clear_ram()` is now a documented no-op — it previously issued `delete_by_query(match_all)` against the durable index a dumped pipeline serves, so `LoggingConfig(clear_ram=True)` emptied it. `OpenSearchBackend.add()` also restores *fit-replaces* semantics, so CV folds and trials no longer accumulate documents and validation folds are no longer retrievable while being scored. `FaissBackend.reset()` now clears `_documents` too, closing a latent index/documents misalignment, and querying an empty OpenSearch index raises an actionable `RuntimeError` instead of a downstream numpy cast error (#345, closes #342).

### Bug fixes

- **Pipelines with tuple-valued config fields reload again.** `Pipeline.dump` and `Context.dump` write `inference_config.yaml` with `yaml.safe_dump`, so tuples are serialized as plain YAML sequences rather than `!!python/tuple` tags that `yaml.safe_load` refuses to construct at load time. Any pipeline using `HashingVectorizerEmbeddingConfig` (whose `ngram_range` is a `tuple[int, int]`) was previously broken end-to-end (#344).

### Dependencies

- **`require(extra)` is now metadata-driven.** The optional-dependency guard reads installed distribution metadata and verifies that *every* dependency of an `autointent` extra — recursively, including nested third-party extras such as `transformers[torch] → accelerate` — is both installed and version-satisfied, raising a single aggregated, actionable `ImportError` instead of a raw deep import failure. `packaging (>=23.2)` is promoted to a core dependency (#339, closes #322).

---

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
