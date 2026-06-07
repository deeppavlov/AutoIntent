"""Tests for the OpenAI embedder backend contract.

Per docs/superpowers/specs/2026-06-07-live-api-test-mocking-strategy.md the
real backend is never exercised in CI; this file pins the shape contract that
FakeOpenaiEmbeddingBackend must uphold so consumers (Embedder, VectorIndex)
keep working when patched.
"""

import numpy as np
import pytest

from autointent.configs import OpenaiEmbeddingConfig, TaskTypeEnum
from tests._fixtures.fake_openai_embedding import FakeOpenaiEmbeddingBackend as OpenaiEmbeddingBackend


@pytest.fixture
def openai_backend_config():
    """Create an OpenAI backend config for testing."""
    return OpenaiEmbeddingConfig(
        model_name="text-embedding-3-small",
        batch_size=2,
        use_cache=False,
        max_retries=1,
        timeout=10.0,
    )


@pytest.fixture
def openai_backend(openai_backend_config: OpenaiEmbeddingConfig):
    """Create an OpenAI backend instance."""
    return OpenaiEmbeddingBackend(openai_backend_config)


class TestOpenaiBackend:
    """Test OpenAI-specific backend functionality."""

    def test_backend_initialization(self, openai_backend: OpenaiEmbeddingBackend):
        """Test backend initialization."""
        assert openai_backend.supports_training is False
        assert openai_backend._client is None  # Client should be lazy-loaded
        assert openai_backend._async_client is None

    def test_client_lazy_loading(self, openai_backend: OpenaiEmbeddingBackend):
        """Test that client is lazy-loaded."""
        assert openai_backend._client is None

        # Client should be loaded on first API call
        embeddings = openai_backend.embed(["Test sentence"])
        assert openai_backend._client is not None
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0

    def test_similarity_calculation(self, openai_backend: OpenaiEmbeddingBackend):
        """Test cosine similarity calculation."""
        embeddings = openai_backend.embed(["Hello", "World", "Hello world"])

        # Test similarity between different embeddings
        similarity = openai_backend.similarity(embeddings[:1], embeddings[1:])

        assert similarity.shape == (1, 2)
        # Cosine similarity should be between -1 and 1
        assert np.all(similarity >= -1.0)
        assert np.all(similarity <= 1.0)

        # "Hello" should be more similar to "Hello world" than to "World"
        hello_to_world = similarity[0, 0]
        hello_to_hello_world = similarity[0, 1]
        assert hello_to_hello_world > hello_to_world

    def test_hash_calculation(self, openai_backend: OpenaiEmbeddingBackend):
        """Test hash calculation for caching."""
        hash1 = openai_backend.get_hash()
        hash2 = openai_backend.get_hash()

        # Hash should be deterministic
        assert hash1 == hash2
        assert isinstance(hash1, int)

    def test_different_models_different_hashes(self):
        """Test that different models produce different hashes."""
        config1 = OpenaiEmbeddingConfig(
            model_name="text-embedding-3-small",
        )
        config2 = OpenaiEmbeddingConfig(
            model_name="text-embedding-ada-002",
        )

        backend1 = OpenaiEmbeddingBackend(config1)
        backend2 = OpenaiEmbeddingBackend(config2)

        assert backend1.get_hash() != backend2.get_hash()

    def test_dimensions_parameter(self):
        """Test that dimensions parameter affects embeddings."""
        # Test with different dimensions (if supported by model)
        config_with_dims = OpenaiEmbeddingConfig(
            model_name="text-embedding-3-small",
            dimensions=512,  # Reduced dimensions
            use_cache=False,
        )

        backend = OpenaiEmbeddingBackend(config_with_dims)
        embeddings = backend.embed(["Test sentence"])

        # Check that embeddings have the specified dimensions
        assert embeddings.shape[1] == 512

    def test_batch_processing(self, openai_backend: OpenaiEmbeddingBackend):
        """Test batch processing functionality."""
        utterances = ["First sentence", "Second sentence", "Third sentence", "Fourth sentence"]

        # Test with batch size 2
        openai_backend.config.batch_size = 2
        embeddings = openai_backend.embed(utterances)

        assert embeddings.shape[0] == 4
        assert embeddings.shape[1] > 0

    def test_async_processing_initialization(self):
        """Test async processing initialization."""
        config = OpenaiEmbeddingConfig(
            model_name="text-embedding-3-small",
            max_concurrent=2,  # Enable async processing
            max_per_second=1.0,
            use_cache=False,
        )

        backend = OpenaiEmbeddingBackend(config)

        # Test async processing
        embeddings = backend.embed(["Test", "async", "processing"])
        assert embeddings.shape[0] == 3

    def test_prompts_application(self):
        """Test that prompts are applied correctly."""
        config = OpenaiEmbeddingConfig(
            model_name="text-embedding-3-small",
            query_prompt="Query:",
            passage_prompt="Passage:",
            use_cache=False,
        )

        backend = OpenaiEmbeddingBackend(config)

        # Test with query task type
        # Get embeddings with and without prompts
        embeddings_no_prompt = backend.embed(["test"], None)
        embeddings_with_prompt = backend.embed(["test"], TaskTypeEnum.query)

        # Embeddings should be different when prompts are applied
        assert not np.allclose(embeddings_no_prompt, embeddings_with_prompt, rtol=1e-3)

    def test_return_tensors_functionality(self, openai_backend: OpenaiEmbeddingBackend):
        """Test return_tensors parameter."""
        utterances = ["Hello world", "Test sentence"]

        # Test numpy return (default)
        embeddings_np = openai_backend.embed(utterances, return_tensors=False)
        assert isinstance(embeddings_np, np.ndarray)

        # Test tensor return
        embeddings_tensor = openai_backend.embed(utterances, return_tensors=True)
        import torch

        assert isinstance(embeddings_tensor, torch.Tensor)

        # Values should be the same
        np.testing.assert_allclose(embeddings_np, embeddings_tensor.cpu().numpy(), rtol=1e-5)
