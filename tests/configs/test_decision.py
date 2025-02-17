import pytest
from pydantic import ValidationError

from autointent.nodes.schemes import OptimizationConfig


@pytest.fixture
def valid_decision_config():
    """Fixture for a valid DecisionNode configuration."""
    return [
        {
            "node_type": "decision",
            "target_metric": "decision_roc_auc",
            "search_space": [
                {"module_name": "argmax"},
                {"module_name": "jinoos", "search_space": [[0.3, 0.5, 0.7]]},
                {"module_name": "threshold", "thresh": [[0.5, 0.6]]},
                {
                    "module_name": "tunable",
                    "n_optuna_trials": [100],
                },
                {"module_name": "adaptive", "search_space": [[0.5]]},
            ],
        }
    ]


def test_valid_decision_config(valid_decision_config):
    """Test that a valid decision config passes validation."""
    config = OptimizationConfig(valid_decision_config)
    assert config[0].node_type == "decision"
    assert config[0].target_metric == "decision_roc_auc"
    assert isinstance(config[0].search_space, list)
    assert config[0].search_space[0].module_name == "argmax"


def test_invalid_decision_config_missing_field():
    """Test that a missing required field raises ValidationError."""
    invalid_config = [
        {
            "node_type": "decision",
            # Missing "target_metric"
            "search_space": [{"module_name": "tunable", "n_optuna_trials": [100]}],
        }
    ]

    with pytest.raises(ValidationError):
        OptimizationConfig(invalid_config)


def test_invalid_decision_config_wrong_type():
    """Test that an invalid field type raises ValidationError."""
    invalid_config = [
        {
            "node_type": "decision",
            "target_metric": "decision_roc_auc",
            "search_space": [
                {
                    "module_name": "threshold",
                    "thresh": ["wrong_type"],  # Should be a list of floats or a single float
                },
                {
                    "module_name": "tunable",
                    "n_optuna_trials": ["not_an_int"],  # Should be an integer
                },
            ],
        }
    ]

    with pytest.raises(ValidationError):
        OptimizationConfig(invalid_config)
