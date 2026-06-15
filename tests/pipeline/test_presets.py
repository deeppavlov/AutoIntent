from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from autointent import Pipeline
from autointent.configs import DataConfig, HPOConfig, LoggingConfig, SentenceTransformerEmbeddingConfig
from autointent.nodes import NodeOptimizer
from tests.conftest import apply_test_models, setup_environment

if TYPE_CHECKING:
    from autointent import Dataset
    from autointent.generation import Generator


@pytest.mark.parametrize(
    "preset",
    [
        "classic-heavy",
        "classic-light",
        "classic-medium",
        "nn-heavy",
        "nn-medium",
        pytest.param("transformers-heavy", marks=pytest.mark.transformers),
        pytest.param("transformers-light", marks=pytest.mark.transformers),
        pytest.param("transformers-no-hpo", marks=pytest.mark.transformers),
        "zero-shot-llm",
        "zero-shot-encoders",
    ],
)
def test_presets(dataset: Dataset, preset: str, patch_llm_scorer_generator: Generator) -> None:
    project_dir = setup_environment()

    pipeline_optimizer = Pipeline.from_preset(preset)  # type: ignore[arg-type]  # reason: parametrize values are runtime strings; mypy can't narrow to the SearchSpacePreset Literal
    apply_test_models(pipeline_optimizer)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(DataConfig(scheme="ho"))
    pipeline_optimizer.set_config(HPOConfig(timeout=60))  # limit budget time because we want tests to be fast

    pipeline_optimizer.fit(dataset, refit_after=False)


def test_apply_test_models_retargets_pipeline_slots() -> None:
    from autointent import Pipeline
    from tests.conftest import (
        TINY_BERT,
        TINY_CROSS_ENCODER,
        TINY_SENTENCE_TRANSFORMER,
        apply_test_models,
    )

    pipeline = Pipeline.from_preset("zero-shot-encoders")
    # Before: zero-shot-encoders sets embedder=multilingual-e5-large-instruct,
    # cross-encoder=bge-reranker-v2-m3.
    apply_test_models(pipeline)

    # apply_test_models() installs a SentenceTransformerEmbeddingConfig (see
    # tests.conftest.tiny_sentence_transformer_config); narrow the EmbedderConfig
    # union to that concrete subclass to access model_name (BaseEmbedderConfig
    # has no model_name).
    assert isinstance(pipeline.embedder_config, SentenceTransformerEmbeddingConfig)
    assert pipeline.embedder_config.model_name == TINY_SENTENCE_TRANSFORMER
    assert pipeline.cross_encoder_config.model_name == TINY_CROSS_ENCODER
    assert pipeline.transformer_config.model_name == TINY_BERT


def test_apply_test_models_rewrites_search_space_bert_entries() -> None:
    from autointent import Pipeline
    from tests.conftest import TINY_BERT, apply_test_models

    pipeline = Pipeline.from_preset("transformers-heavy")
    # Before: search_space has module_name='bert' with
    # classification_model_config: [{model_name: 'microsoft/deberta-v3-large'}]
    apply_test_models(pipeline)

    # Pipeline.from_preset() returns an optimization-mode Pipeline whose nodes
    # are NodeOptimizer; the typed union with InferenceNode does not expose
    # modules_search_spaces. Assert each node is a NodeOptimizer to access it.
    bert_entries: list[dict[str, Any]] = []
    for node in pipeline.nodes.values():
        assert isinstance(node, NodeOptimizer)
        bert_entries.extend(entry for entry in node.modules_search_spaces if entry.get("module_name") == "bert")
    assert bert_entries, "transformers-heavy preset must have a bert module entry"

    for entry in bert_entries:
        cmc = entry.get("classification_model_config")
        # The field is a list of dicts in YAML (Optuna categorical search space).
        assert isinstance(cmc, list), f"unexpected shape: {cmc!r}"
        assert cmc, f"unexpected shape: {cmc!r}"
        for cfg in cmc:
            assert cfg.get("model_name") == TINY_BERT, (
                f"search-space bert.classification_model_config.model_name must be "
                f"retargeted to {TINY_BERT}; got {cfg.get('model_name')!r}"
            )


def test_apply_test_models_drops_stale_revision_in_search_space() -> None:
    """When a search-space entry pins model_name AND revision (e.g. catboost
    in tests/assets/configs/multiclass.yaml), the walker rewrites the
    model_name but must also drop the now-wrong revision so the
    HFModelConfig validator refills it from DEFAULT_REVISIONS.
    """
    from autointent import Pipeline
    from tests.conftest import apply_test_models, get_search_space

    pipeline = Pipeline.from_search_space(get_search_space("multiclass"))
    apply_test_models(pipeline)

    for node in pipeline.nodes.values():
        # Pipeline.from_search_space() returns an optimization-mode Pipeline
        # whose nodes are NodeOptimizer (see test_apply_test_models_rewrites_…).
        assert isinstance(node, NodeOptimizer)
        for entry in node.modules_search_spaces:
            for field in ("classification_model_config", "embedder_config", "cross_encoder_config"):
                value = entry.get(field)
                if isinstance(value, list):
                    for cfg in value:
                        if isinstance(cfg, dict) and "revision" in cfg:
                            msg = (
                                f"{field!r} entry still has a stale revision after retarget: {cfg!r}. "
                                "_rewrite_field must pop revision when it rewrites model_name."
                            )
                            raise AssertionError(msg)
