import pytest
from pydantic import ValidationError

from autointent.nodes import OptimizationConfig


@pytest.fixture
def valid_embedding_config():
    """Fixture for a valid EmbeddingNode configuration."""
    return [
        {
            "node_type": "embedding",
            "target_metric": "retrieval_mrr",
            "search_space": [
                {"module_name": "logreg_embedding", "embedder_config": ["sergeyzh/rubert-tiny-turbo"], "cv": [3, 5]},
                {
                    "module_name": "retrieval",
                    "embedder_config": ["sentence-transformers/all-MiniLM-L6-v2"],
                    "k": [5, 10],
                },
            ],
        }
    ]


def test_valid_embedding_config(valid_embedding_config):
    """Test that a valid embedding config passes validation."""
    config = OptimizationConfig(valid_embedding_config)
    assert config[0].node_type == "embedding"
    assert config[0].target_metric == "retrieval_mrr"
    assert isinstance(config[0].search_space, list)
    assert config[0].search_space[0].module_name == "logreg_embedding"
    assert "embedder_config" in config[0].search_space[0].model_dump()


def test_invalid_embedding_config_missing_field():
    """Test that a missing required field raises ValidationError."""
    invalid_config = [
        {
            "node_type": "embedding",
            # Missing "target_metric"
            "search_space": [
                {
                    "module_name": "retrieval",
                    "embedder_config": ["sentence-transformers/all-MiniLM-L6-v2"],
                    "k": [5, 10],
                }
            ],
        }
    ]

    with pytest.raises(ValidationError):
        OptimizationConfig(invalid_config)


def test_invalid_embedding_config_wrong_type():
    """Test that an invalid field type raises ValidationError."""
    invalid_config = [
        {
            "node_type": "embedding",
            "target_metric": "retrieval_mrr",
            "search_space": [
                {
                    "module_name": "logreg_embedding",
                    "embedder_config": "not_a_list",  # Should be a list of strings
                    "cv": ["wrong_type"],  # Should be a list of integers
                }
            ],
        }
    ]

    with pytest.raises(ValidationError):
        OptimizationConfig(invalid_config)
