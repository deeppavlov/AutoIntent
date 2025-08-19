import numpy as np
import pytest

from autointent._wrappers.embedder import Embedder
from autointent.configs import EmbedderConfig, TaskTypeEnum


@pytest.fixture
def prompt_embedder_config():
    """Create embedder config with different prompts."""
    return EmbedderConfig(
        model_name="sergeyzh/rubert-tiny-turbo",
        batch_size=4,
        device="cpu",
        use_cache=False,
        default_prompt="Represent this text:",
        query_prompt="Search query:",
        passage_prompt="Document:",
        classification_prompt="Classify:",
    )


def test_different_task_prompts(prompt_embedder_config):
    """Test that different task types produce different embeddings."""
    embedder = Embedder(prompt_embedder_config)
    test_utterance = ["Test sentence"]

    default_emb = embedder.embed(test_utterance, TaskTypeEnum.default)
    query_emb = embedder.embed(test_utterance, TaskTypeEnum.query)
    passage_emb = embedder.embed(test_utterance, TaskTypeEnum.passage)
    classification_emb = embedder.embed(test_utterance, TaskTypeEnum.classification)

    # Different prompts should produce different embeddings
    assert not np.allclose(default_emb, query_emb, rtol=1e-3)
    assert not np.allclose(default_emb, passage_emb, rtol=1e-3)
    assert not np.allclose(default_emb, classification_emb, rtol=1e-3)


def test_fallback_to_default_prompt():
    """Test fallback to default prompt when specific prompt not set."""
    config = EmbedderConfig(
        model_name="sergeyzh/rubert-tiny-turbo",
        default_prompt="Default:",
        use_cache=False,
    )
    embedder = Embedder(config)

    # Should use default prompt when specific task prompt not available
    embeddings1 = embedder.embed(["test"], TaskTypeEnum.cluster)
    embeddings2 = embedder.embed(["test"], TaskTypeEnum.default)

    np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5)
