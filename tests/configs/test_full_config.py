import pytest
from pydantic import ValidationError

from autointent.nodes.schemes import OptimizationConfig
from tests.conftest import get_search_space


def test_validate_full_config():
    config = get_search_space("full_training")
    validated_config = OptimizationConfig(**config)
    assert isinstance(validated_config, OptimizationConfig)


def test_not_valid_reporting():
    config = get_search_space("full_training")
    config["logging_config"]["report_to"] = "test"

    with pytest.raises(ValidationError):
        OptimizationConfig(**config)
