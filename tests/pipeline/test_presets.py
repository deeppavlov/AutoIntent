import os

import pytest

from autointent import Pipeline
from autointent.configs import DataConfig, HPOConfig, LoggingConfig
from tests.conftest import apply_test_models, setup_environment


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
def test_presets(dataset, preset):
    project_dir = setup_environment()

    pipeline_optimizer = Pipeline.from_preset(preset)
    apply_test_models(pipeline_optimizer)

    if preset == "zero-shot-llm" and not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_MODEL_NAME")):
        pytest.skip(reason="OpenAI API key or model name is missing.")

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(DataConfig(scheme="ho"))
    pipeline_optimizer.set_config(HPOConfig(timeout=60))  # limit budget time because we want tests to be fast

    pipeline_optimizer.fit(dataset, refit_after=False)


def test_apply_test_models_retargets_pipeline_slots():
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

    assert pipeline.embedder_config.model_name == TINY_SENTENCE_TRANSFORMER
    assert pipeline.cross_encoder_config.model_name == TINY_CROSS_ENCODER
    assert pipeline.transformer_config.model_name == TINY_BERT


def test_apply_test_models_rewrites_search_space_bert_entries():
    from autointent import Pipeline
    from tests.conftest import TINY_BERT, apply_test_models

    pipeline = Pipeline.from_preset("transformers-heavy")
    # Before: search_space has module_name='bert' with
    # classification_model_config: [{model_name: 'microsoft/deberta-v3-large'}]
    apply_test_models(pipeline)

    bert_entries = [
        entry
        for node in pipeline.nodes.values()
        for entry in node.modules_search_spaces
        if entry.get("module_name") == "bert"
    ]
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
