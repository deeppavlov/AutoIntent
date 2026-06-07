"""Pinned commit SHAs for Hugging Face models used as defaults in autointent.

This module is the canonical source of truth for every SHA pin in the
project. The dict is consumed by:

  - autointent.configs._transformers.HFModelConfig._apply_default_revision,
    which auto-fills `revision` on configs whose model_name is a key here.
    _transformers.py also re-exports DEFAULT_REVISIONS for backward
    compatibility; do not remove that re-export without updating consumers.
  - .ci/warm_hf_cache.py, which joins repo IDs from the prewarm YAML
    against this dict to produce pinned entries for the cache warmer.
    warm_hf_cache loads this file via importlib.util.spec_from_file_location
    (bypassing autointent's package init) so it works in the slim
    warm-cache CI env that has no pydantic/numpy/etc.

LEAF MODULE INVARIANT: this file must contain only `from __future__`
imports (no other imports of any kind). spec_from_file_location does
execute regular imports if added, so non-__future__ imports would couple
the warm-cache job's slim env to whatever those transitive imports need.
Keep the surface zero so the contract stays obvious: this file is data.

Two unit tests in tests/configs/test_combined_config.py enforce the
contract:
  - test_pinned_revisions_module_has_no_runtime_imports (AST structural)
  - test_leaf_module_loadable_without_autointent_package (hermetic load
    via subprocess; the actual mechanism the warm-cache job uses)
Do not relax these tests to add an import.

Update an entry below when you intentionally want to move a default to a
newer revision. To add a new pinned model, add a new entry here, then
(optionally) list its repo ID in .ci/hf-prewarm-linux.yaml /
hf-prewarm-windows.yaml if it should be CI-prewarmed.
"""

from __future__ import annotations

DEFAULT_REVISIONS: dict[str, str] = {
    "prajjwal1/bert-tiny": "79779625a0a40f1eee8496e16056bc0d7766df22",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "intfloat/multilingual-e5-large-instruct": "274baa43b0e13e37fafa6428dbc7938e62e5c439",
    "intfloat/multilingual-e5-small": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
    "avsolatorio/GIST-small-Embedding-v0": "75e62fd210b9fde790430e0b2f040b0b00a021b1",
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    # Used heavily in the embedder test suite (tests/embedder/conftest.py).
    # Pinning the SHA here lets HFModelConfig auto-fill it via the validator
    # so sentence-transformers never asks the Hub for "main" — which 429s
    # under parallel matrix load even when the model files are cached.
    "sergeyzh/rubert-tiny-turbo": "93769a3baad2b037e5c2e4312fccf6bcfe082bf1",
    "microsoft/deberta-v3-large": "64a8c8eab3e352a784c658aef62be1662607476f",
    "microsoft/deberta-v3-small": "a36c739020e01763fe789b4b85e2df55d6180012",
}
