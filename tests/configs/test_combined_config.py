import pytest
from pydantic import ValidationError

from autointent.nodes.schemes import (
    OptimizationSearchSpaceConfig,
)
from tests.conftest import get_search_space


@pytest.fixture
def valid_optimizer_config():
    """Fixture for a valid OptimizerConfig."""
    return [
        {
            "node_type": "scoring",
            "target_metric": "scoring_roc_auc",
            "search_space": [
                {
                    "module_name": "dnnc",
                    "cross_encoder_config": [
                        {"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2", "train_head": True},
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


def test_valid_optimizer_config(valid_optimizer_config):
    """Test that a valid optimizer config passes validation."""
    config = OptimizationSearchSpaceConfig(valid_optimizer_config)
    assert config[0].node_type == "scoring"
    assert config[1].node_type == "embedding"


@pytest.mark.parametrize(
    "task_type",
    ["multiclass", "multilabel", "description"],
)
def test_optimizer_config(task_type):
    search_space = get_search_space(task_type)
    config = OptimizationSearchSpaceConfig(search_space)
    assert config


def test_invalid_optimizer_config_missing_field():
    """Test that a missing required field raises ValidationError."""
    invalid_config = [
        {
            "node_type": "scoring",
            # Missing "target_metric"
            "search_space": [
                {"module_name": "dnnc", "cross_encoder_name": ["cross-encoder/ms-marco-MiniLM-L-6-v2"], "k": [1, 3]}
            ],
        }
    ]

    with pytest.raises(ValidationError):
        OptimizationSearchSpaceConfig(invalid_config)


def test_invalid_optimizer_config_wrong_type():
    """Test that an invalid field type raises ValidationError."""
    invalid_config = [
        {
            "node_type": "scoring",
            "target_metric": "scoring_roc_auc",
            "search_space": [
                {
                    "module_name": "dnnc",
                    "cross_encoder_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Should be a list
                    "k": "wrong_type",  # Should be a list of integers
                    "train_head": "true",  # Should be a boolean, not a string
                }
            ],
        }
    ]

    with pytest.raises(ValidationError):
        OptimizationSearchSpaceConfig(invalid_config)
