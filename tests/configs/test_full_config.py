import pytest
from pydantic import ValidationError

from autointent import OptimizationConfig
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


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_seed_accepts_non_negative(seed: int) -> None:
    config = OptimizationConfig(seed=seed, search_space=[])
    assert config.seed == seed


def test_seed_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        OptimizationConfig(seed=-1, search_space=[])
