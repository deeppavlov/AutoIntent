import numpy as np
import pytest

from autointent._wrappers.embedder import Embedder
from autointent.configs._transformers import EmbedderConfig


@pytest.fixture
def simple_embedder_config():
    """Create a simple embedder config for testing."""
    return EmbedderConfig(
        model_name="sergeyzh/rubert-tiny-turbo",
        batch_size=4,
        device="cpu",
        use_cache=False,
    )


def test_embedding_calculation(simple_embedder_config):
    """Test basic embedding calculation functionality."""
    embedder = Embedder(simple_embedder_config)
    test_utterances = ["Hello world", "Test sentence", "Another example"]

    embeddings = embedder.embed(test_utterances)

    assert embeddings.shape[0] == len(test_utterances)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)  # normalized


def test_embedding_reproducibility(simple_embedder_config):
    """Test that embeddings are reproducible for same input."""
    embedder = Embedder(simple_embedder_config)
    test_utterances = ["Hello world", "Test sentence"]

    embeddings1 = embedder.embed(test_utterances)
    embeddings2 = embedder.embed(test_utterances)

    np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5)


def test_single_utterance(simple_embedder_config):
    """Test embedding calculation for single utterance."""
    embedder = Embedder(simple_embedder_config)

    embeddings = embedder.embed(["Single test sentence"])
    assert embeddings.shape[0] == 1
    assert np.allclose(np.linalg.norm(embeddings[0]), 1.0, atol=1e-5)


def test_similarity_symmetry():
    """Test that similarity is symmetric for cosine similarity."""
    config = EmbedderConfig(model_name="sergeyzh/rubert-tiny-turbo", similarity_fn_name="cosine", use_cache=False)
    embedder = Embedder(config)

    utterances = ["Hello world", "Test sentence"]
    embeddings = embedder.embed(utterances)

    sim1 = embedder.similarity(embeddings[:1], embeddings[1:])
    sim2 = embedder.similarity(embeddings[1:], embeddings[:1])

    np.testing.assert_allclose(sim1, sim2.T, rtol=1e-5)
