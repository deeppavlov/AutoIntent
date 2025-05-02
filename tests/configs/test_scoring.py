import pytest

from autointent.nodes import NodeOptimizer


@pytest.fixture
def valid_scoring_config():
    """Fixture for a valid ScoringNode configuration."""
    return {
        "node_type": "scoring",
        "target_metric": "scoring_roc_auc",
        "search_space": [
            {
                "module_name": "dnnc",
                "cross_encoder_config": [
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    {"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2", "train_head": True},
                ],
                "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
                "k": [5, 10],
            },
            {
                "module_name": "knn",
                "embedder_config": ["sentence-transformers/all-MiniLM-L6-v2"],
                "k": [5, 10],
                "weights": ["uniform", "distance"],
            },
            {"module_name": "linear", "embedder_config": ["sergeyzh/rubert-tiny-turbo"], "cv": [3, 5]},
            {
                "module_name": "mlknn",
                "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
                "k": [5, 10],
                "s": [1.0, 0.5],
                "ignore_first_neighbours": [0, 1],
            },
            {
                "module_name": "description",
                "embedder_config": ["sentence-transformers/all-MiniLM-L6-v2"],
                "temperature": [0.5, 1.0],
            },
            {
                "module_name": "rerank",
                "cross_encoder_config": ["cross-encoder/ms-marco-MiniLM-L-6-v2"],
                "embedder_config": ["sergeyzh/rubert-tiny-turbo"],
                "k": [5],
                "weights": ["distance"],
                "use_crosencoder_scores": [True, False],
            },
            {
                "module_name": "sklearn",
                "embedder_config": ["sentence-transformers/all-MiniLM-L6-v2"],
                "clf_name": ["LogisticRegression"],
                "C": [0.2, 0.3],
            },
        ],
    }


def test_valid_scoring_config(valid_scoring_config):
    """Test that a valid scoring config passes validation."""
    node = NodeOptimizer(**valid_scoring_config)
    assert node.node_type == "scoring"
    assert node.target_metric == "scoring_roc_auc"
    assert isinstance(node.modules_search_spaces, list)
    assert node.modules_search_spaces[0]["module_name"] == "dnnc"


def test_invalid_scoring_config_missing_field():
    """Test that a missing required field raises ValidationError."""
    invalid_config = {
        "node_type": "scoring",
        # Missing "target_metric"
        "search_space": [
            {"module_name": "dnnc", "cross_encoder_name": ["cross-encoder/ms-marco-MiniLM-L-6-v2"], "k": [5, 10]}
        ],
    }

    with pytest.raises(TypeError):
        NodeOptimizer(*invalid_config)


def test_invalid_scoring_config_wrong_type():
    """Test that an invalid field type raises ValidationError."""
    invalid_config = {
        "node_type": "scoring",
        "target_metric": "scoring_roc_auc",
        "search_space": [
            {
                "module_name": "knn",
                "embedder_config": "should_be_list",  # Should be a list of strings
                "k": "not_an_int_list",  # Should be a list of integers
                "weights": ["uniform", "distance"],
            }
        ],
    }

    with pytest.raises(TypeError):
        NodeOptimizer(**invalid_config)
