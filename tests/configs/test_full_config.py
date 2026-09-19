import pytest
from pydantic import ValidationError

from autointent import OptimizationConfig, Pipeline
from tests.conftest import load_optimization_config


def test_validate_full_config() -> None:
    config = load_optimization_config("full_training")
    validated_config = OptimizationConfig(**config)
    assert isinstance(validated_config, OptimizationConfig)


def test_not_valid_reporting() -> None:
    config = load_optimization_config("full_training")
    config["logging_config"]["report_to"] = "test"

    with pytest.raises(ValidationError):
        OptimizationConfig(**config)


@pytest.mark.parametrize("seed", [0, 42])
def test_non_negative_seed(seed: int) -> None:
    config = OptimizationConfig(seed=seed, search_space=[])
    assert config.seed == seed


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValidationError, match="seed"):
        OptimizationConfig(seed=-1, search_space=[])


def test_pipeline_from_preset_accepts_zero_seed() -> None:
    pipeline = Pipeline.from_preset("classic-light", seed=0)
    assert pipeline._seed == 0


def test_pipeline_from_optimization_config_accepts_zero_seed() -> None:
    config = load_optimization_config("full_training")
    config["seed"] = 0
    pipeline = Pipeline.from_optimization_config(config)
    assert pipeline._seed == 0
