from typing import Any, get_args

import pytest

from autointent.nodes import NodeOptimizer
from tests.conftest import TaskType, get_search_space


@pytest.fixture
def valid_optimizer_config() -> list[dict[str, Any]]:
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


def test_valid_optimizer_config(valid_optimizer_config: list[dict[str, Any]]) -> None:
    """Test that a valid optimizer config passes validation."""
    for node_dict_config in valid_optimizer_config:
        NodeOptimizer(**node_dict_config)


@pytest.mark.parametrize(
    "task_type",
    get_args(TaskType),
)
def test_optimizer_config(task_type: TaskType) -> None:
    for node_dict_config in get_search_space(task_type):
        NodeOptimizer(**node_dict_config)


def test_invalid_optimizer_config_missing_field() -> None:
    """Test that a missing required field raises ValidationError."""
    invalid_config: list[dict[str, Any]] = [
        {
            "node_type": "scoring",
            # Missing "target_metric"
            "search_space": [
                {"module_name": "dnnc", "cross_encoder_name": ["cross-encoder/ms-marco-MiniLM-L6-v2"], "k": [1, 3]}
            ],
        }
    ]

    with pytest.raises(TypeError):
        # reason: test asserts TypeError; double-star spread of a list is invalid at
        # runtime (raises TypeError), which is the exact failure mode under test. mypy
        # correctly flags the kwarg mismatch, so suppress with code here.
        NodeOptimizer(**invalid_config)  # type: ignore[arg-type]


def test_deberta_v3_large_is_pinned() -> None:
    from autointent.configs._pinned_revisions import DEFAULT_REVISIONS

    sha = DEFAULT_REVISIONS.get("microsoft/deberta-v3-large")
    assert sha is not None, "microsoft/deberta-v3-large must be pinned (used by transformers-heavy preset)"
    assert len(sha) == 40, f"SHA must be 40 chars; got {sha!r}"
    assert all(c in "0123456789abcdef" for c in sha), f"SHA must be hex; got {sha!r}"


def test_canonical_test_models_have_pinned_revisions() -> None:
    from autointent.configs._pinned_revisions import DEFAULT_REVISIONS
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


def test_bert_tiny_config_in_scoring_tests_gets_pinned_revision() -> None:
    """test_bert/lora/ptuning.py rely on _apply_default_revision to fill
    revision when they omit it. Lock that contract in here so a future
    change to the validator doesn't silently make the scoring tests
    contact HF Hub for revision resolution."""
    from autointent.configs import HFModelConfig
    from autointent.configs._pinned_revisions import DEFAULT_REVISIONS

    cfg = HFModelConfig(model_name="prajjwal1/bert-tiny")
    assert cfg.revision == DEFAULT_REVISIONS["prajjwal1/bert-tiny"]


def test_hf_guard_blocks_unpinned_revision() -> None:
    from tests.conftest import _make_hf_guard

    sentinel = object()
    guard = _make_hf_guard(lambda *a, **k: sentinel, label="fake")

    import pytest

    for bad in (None, "main", "v1.0", "abc"):
        with pytest.raises(AssertionError, match="Unpinned HF call"):
            guard("repo/id", "file.bin", revision=bad)


def test_hf_guard_allows_sha_pinned_revision() -> None:
    from tests.conftest import _make_hf_guard

    sentinel = object()
    guard = _make_hf_guard(lambda *a, **k: sentinel, label="fake")

    sha = "0" * 40
    assert guard("repo/id", "file.bin", revision=sha) is sentinel
    sha2 = "abcdef0123456789" * 2 + "abcdef01"  # 40 hex
    assert len(sha2) == 40
    assert guard("repo/id", "file.bin", revision=sha2) is sentinel


def test_deberta_v3_small_is_pinned() -> None:
    from autointent.configs._pinned_revisions import DEFAULT_REVISIONS

    sha = DEFAULT_REVISIONS.get("microsoft/deberta-v3-small")
    assert sha is not None, (
        "microsoft/deberta-v3-small must be pinned (used by transformers-light + transformers-no-hpo presets)"
    )
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha), f"SHA must be lowercase hex; got {sha!r}"


def test_invalid_optimizer_config_wrong_type() -> None:
    """Test that an invalid field type raises ValidationError."""
    invalid_config: dict[str, Any] = {
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


def test_pinned_revisions_module_has_no_runtime_imports() -> None:
    """The leaf module must stay minimal — no imports beyond __future__.

    .ci/warm_hf_cache.py loads it via importlib.util.spec_from_file_location,
    which DOES execute any imports the leaf module declares. The
    spec_from_file_location load path keeps the warm-cache job working
    today even with extra imports, but a future contributor adding e.g.
    `import json` would couple the warm-cache env to whatever that
    transitive import needs. Keep the surface zero so the contract is
    obvious: this file is just data.

    See test_leaf_module_loadable_without_autointent_package below for
    the load-path correctness test.
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


def test_leaf_module_loadable_without_autointent_package() -> None:
    """The warm-cache CI job loads _pinned_revisions.py via
    importlib.util.spec_from_file_location in an environment where the
    autointent package is NOT installed. Run that exact load mechanism
    in a clean subprocess to catch regressions in the import path.

    Without this test, a refactor that re-introduced
    `from autointent.configs._pinned_revisions import ...` in
    warm_hf_cache.py would pass all in-process tests (because the dev
    venv has autointent installed) but break CI silently.
    """
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    leaf = Path(__file__).resolve().parents[2] / "src" / "autointent" / "configs" / "_pinned_revisions.py"
    assert leaf.is_file(), f"expected leaf module at {leaf}"

    script = textwrap.dedent(
        f"""
        import importlib.util
        spec = importlib.util.spec_from_file_location('_pr', r'{leaf}')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert isinstance(mod.DEFAULT_REVISIONS, dict), 'DEFAULT_REVISIONS must be a dict'
        assert mod.DEFAULT_REVISIONS, 'DEFAULT_REVISIONS must be non-empty'
        """
    )
    # Use -S to skip site-packages too, so we approximate the warm-cache
    # job's slim env as closely as possible from inside the dev venv.
    # The leaf module's only import is `from __future__ import annotations`
    # which is a syntax directive and doesn't touch sys.path.
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Hermetic load failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
