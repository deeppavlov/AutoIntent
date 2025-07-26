import os

import pytest

from autointent import Pipeline
from autointent.configs import DataConfig, HPOConfig, LoggingConfig
from tests.conftest import setup_environment


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

    if preset == "zero-shot-llm" and not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_MODEL_NAME")):
        pytest.skip(reason="OpenAI API key or model name is missing.")

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(DataConfig(scheme="ho"))
    pipeline_optimizer.set_config(HPOConfig(timeout=60))  # limit budget time because we want tests to be fast

    pipeline_optimizer.fit(dataset, refit_after=False)
