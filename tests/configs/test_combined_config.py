from typing import get_args

import pytest

from autointent.nodes import NodeOptimizer
from tests.conftest import TaskType, get_search_space


@pytest.fixture
def valid_optimizer_config():
    """Fixture for a valid OptimizerConfig."""
    return [
        {
            "node_type": "scoring",
            "target_metric": "scoring_roc_auc",
            "search_space": [
                {
                    "module_name": "dnnc",
                    "cross_encoder_config": [
                        {"model_name": "cross-encoder/ms-marco-MiniLM-L6-v2", "train_head": True},
                        {"model_name": "avsolatorio/GIST-small-Embedding-v0", "train_head": False},
                    ],
                    "k": [1, 3],
                }
            ],
        },
        {
            "node_type": "embedding",
            "target_metric": "retrieval_hit_rate",
            "search_space": [
                {
                    "module_name": "retrieval",
                    "k": [5, 10],
                    "embedder_config": [
                        "sentence-transformers/all-MiniLM-L6-v2",
                        "avsolatorio/GIST-small-Embedding-v0",
                    ],
                }
            ],
        },
    ]


def test_valid_optimizer_config(valid_optimizer_config):
    """Test that a valid optimizer config passes validation."""
    for node_dict_config in valid_optimizer_config:
        NodeOptimizer(**node_dict_config)


@pytest.mark.parametrize(
    "task_type",
    get_args(TaskType),
)
def test_optimizer_config(task_type):
    for node_dict_config in get_search_space(task_type):
        NodeOptimizer(**node_dict_config)


def test_invalid_optimizer_config_missing_field():
    """Test that a missing required field raises ValidationError."""
    invalid_config = [
        {
            "node_type": "scoring",
            # Missing "target_metric"
            "search_space": [
                {"module_name": "dnnc", "cross_encoder_name": ["cross-encoder/ms-marco-MiniLM-L6-v2"], "k": [1, 3]}
            ],
        }
    ]

    with pytest.raises(TypeError):
        NodeOptimizer(**invalid_config)


def test_deberta_v3_large_is_pinned():
    from autointent.configs._transformers import DEFAULT_REVISIONS

    sha = DEFAULT_REVISIONS.get("microsoft/deberta-v3-large")
    assert sha is not None, "microsoft/deberta-v3-large must be pinned (used by transformers-heavy preset)"
    assert len(sha) == 40, f"SHA must be 40 chars; got {sha!r}"
    assert all(c in "0123456789abcdef" for c in sha), f"SHA must be hex; got {sha!r}"


def test_canonical_test_models_have_pinned_revisions():
    from autointent.configs._transformers import DEFAULT_REVISIONS
    from tests.conftest import (
        TINY_BERT,
        TINY_CROSS_ENCODER,
        TINY_SENTENCE_TRANSFORMER,
        tiny_bert_config,
        tiny_cross_encoder_config,
        tiny_sentence_transformer_config,
    )

    for name in (TINY_BERT, TINY_CROSS_ENCODER, TINY_SENTENCE_TRANSFORMER):
        sha = DEFAULT_REVISIONS.get(name)
        assert sha is not None, f"{name} must be pinned in DEFAULT_REVISIONS"
        assert len(sha) == 40, f"{name} SHA must be 40 chars; got {sha!r}"

    # Each helper returns a config whose validator filled in the SHA.
    assert tiny_bert_config().revision == DEFAULT_REVISIONS[TINY_BERT]
    assert tiny_cross_encoder_config().revision == DEFAULT_REVISIONS[TINY_CROSS_ENCODER]
    assert tiny_sentence_transformer_config().model_name == TINY_SENTENCE_TRANSFORMER


def test_hf_guard_blocks_unpinned_revision():
    from tests.conftest import _make_hf_guard

    sentinel = object()
    guard = _make_hf_guard(lambda *a, **k: sentinel, label="fake")

    import pytest

    for bad in (None, "main", "v1.0", "abc"):
        with pytest.raises(AssertionError, match="Unpinned HF call"):
            guard("repo/id", "file.bin", revision=bad)


def test_hf_guard_allows_sha_pinned_revision():
    from tests.conftest import _make_hf_guard

    sentinel = object()
    guard = _make_hf_guard(lambda *a, **k: sentinel, label="fake")

    sha = "0" * 40
    assert guard("repo/id", "file.bin", revision=sha) is sentinel
    sha2 = "abcdef0123456789" * 2 + "abcdef01"  # 40 hex
    assert len(sha2) == 40
    assert guard("repo/id", "file.bin", revision=sha2) is sentinel


def test_deberta_v3_small_is_pinned():
    from autointent.configs._transformers import DEFAULT_REVISIONS

    sha = DEFAULT_REVISIONS.get("microsoft/deberta-v3-small")
    assert sha is not None, (
        "microsoft/deberta-v3-small must be pinned (used by transformers-light + transformers-no-hpo presets)"
    )
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha), f"SHA must be lowercase hex; got {sha!r}"


def test_invalid_optimizer_config_wrong_type():
    """Test that an invalid field type raises ValidationError."""
    invalid_config = {
        "node_type": "scoring",
        "target_metric": "scoring_roc_auc",
        "search_space": [
            {
                "module_name": "dnnc",
                "cross_encoder_name": "cross-encoder/ms-marco-MiniLM-L6-v2",  # Should be a list
                "k": "wrong_type",  # Should be a list of integers
                "train_head": "true",  # Should be a boolean, not a string
            }
        ],
    }

    with pytest.raises(TypeError):
        NodeOptimizer(**invalid_config)


def test_pinned_revisions_module_has_no_runtime_imports():
    """The leaf module must be loadable without autointent's deps installed.

    .ci/warm_hf_cache.py imports it via a sys.path shim that does NOT
    install pydantic, datasets, or any other autointent dep. If a future
    edit adds e.g. `import json` to _pinned_revisions, the warm-cache job
    silently keeps working in environments that happen to have json
    available but breaks in stricter ones; this test prevents that drift.
    """
    import ast
    from pathlib import Path

    from autointent.configs import _pinned_revisions

    source = Path(_pinned_revisions.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module == "__future__", (
                f"_pinned_revisions.py must not import {node.module!r} "
                f"(only `from __future__ import ...` is allowed)"
            )
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
            assert not modules, (
                f"_pinned_revisions.py must not contain `import` statements; "
                f"found: {modules}"
            )
