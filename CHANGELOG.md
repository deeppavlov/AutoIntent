# Changelog

All notable changes to this project are documented in this file. Release notes are grouped by theme rather than listing every commit.

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
