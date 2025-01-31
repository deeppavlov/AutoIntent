import pytest
from pydantic import ValidationError

from autointent.nodes import OptimizationConfig


@pytest.fixture
def valid_regexp_config():
    """Fixture for a valid RegExp node configuration."""
    return [
        {"node_type": "regexp", "target_metric": "regexp_partial_accuracy", "search_space": [{"module_name": "regexp"}]}
    ]


def test_valid_regexp_config(valid_regexp_config):
    """Test that a valid RegExp config passes validation."""
    config = OptimizationConfig(valid_regexp_config)
    assert config[0].node_type == "regexp"
    assert config[0].target_metric == "regexp_partial_accuracy"
    assert isinstance(config[0].search_space, list)
    assert config[0].search_space[0].module_name == "regexp"


def test_invalid_regexp_config_missing_field():
    """Test that a missing required field raises ValidationError."""
    invalid_config = {
        "node_type": "regexp",
        # Missing "target_metric"
        "search_space": [{"module_name": "regexp"}],
    }

    with pytest.raises(ValidationError):
        OptimizationConfig(invalid_config)


def test_invalid_regexp_config_wrong_type():
    """Test that an invalid field type raises ValidationError."""
    invalid_config = {
        "node_type": "regexp",
        "target_metric": "regexp_partial_accuracy",
        "search_space": "should_be_a_list",  # Should be a list of dicts
    }

    with pytest.raises(ValidationError):
        OptimizationConfig(invalid_config)
