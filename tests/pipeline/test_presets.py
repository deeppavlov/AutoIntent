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
        pytest.param(
            "zero-shot-llm",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "LLMDescriptionScorer.dump/load drops generator_config; "
                    "preset's dump_modules+clear_ram cycle then fails on predict. See "
                    "https://github.com/deeppavlov/AutoIntent/issues/299. Flip when fixed."
                ),
            ),
        ),
        "zero-shot-encoders",
    ],
)
def test_presets(dataset, preset, patch_llm_scorer_generator):
    project_dir = setup_environment()

    pipeline_optimizer = Pipeline.from_preset(preset)

    pipeline_optimizer.set_config(LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True))
    pipeline_optimizer.set_config(DataConfig(scheme="ho"))
    pipeline_optimizer.set_config(HPOConfig(timeout=60))  # limit budget time because we want tests to be fast

    pipeline_optimizer.fit(dataset, refit_after=False)
