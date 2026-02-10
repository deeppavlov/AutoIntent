from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from autointent._wrappers.embedder import Embedder

from .conftest import backend_configs

if TYPE_CHECKING:
    from autointent.configs import EmbedderConfig


@pytest.mark.parametrize("embedder_config", backend_configs)
class TestEmbedderBasic:
    """Unified test class for Embedder with different backends."""

    @pytest.fixture
    def embedder(self, embedder_config: EmbedderConfig) -> Embedder:
        """Create an Embedder instance for testing."""
        return Embedder(embedder_config)

    def test_embedding_calculation(self, embedder: Embedder):
        """Test basic embedding calculation functionality."""
        test_utterances = ["Hello world", "Test sentence", "Another example"]

        embeddings = embedder.embed(test_utterances)

        assert embeddings.shape[0] == len(test_utterances)
        assert embeddings.shape[1] > 0  # Should have some dimensions
        # Note: OpenAI embeddings may not be normalized by default, so we check for SentenceTransformers only
        if hasattr(embedder.config, "similarity_fn_name"):
            assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)  # normalized

    def test_embedding_reproducibility(self, embedder: Embedder):
        """Test that embeddings are reproducible for same input."""
        test_utterances = ["Hello world", "Test sentence"]

        embeddings1 = embedder.embed(test_utterances)
        embeddings2 = embedder.embed(test_utterances)

        np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5)

    def test_single_utterance(self, embedder: Embedder):
        """Test embedding calculation for single utterance."""
        embeddings = embedder.embed(["Single test sentence"])
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0

    def test_similarity_calculation(self, embedder: Embedder):
        """Test similarity calculation between embeddings."""
        utterances = ["Hello world", "Test sentence", "Another test"]
        embeddings = embedder.embed(utterances)

        # Test similarity between first and second embedding
        sim_matrix = embedder.similarity(embeddings[:1], embeddings[1:])

        assert sim_matrix.shape == (1, 2)
        # Similarity should be between -1 and 1 for cosine similarity
        assert np.all(sim_matrix >= -1.0)
        assert np.all(sim_matrix <= 1.0)

    def test_similarity_symmetry(self, embedder: Embedder):
        """Test that similarity is symmetric."""
        utterances = ["Hello world", "Test sentence"]
        embeddings = embedder.embed(utterances)

        sim1 = embedder.similarity(embeddings[:1], embeddings[1:])
        sim2 = embedder.similarity(embeddings[1:], embeddings[:1])

        np.testing.assert_allclose(sim1, sim2.T, rtol=1e-5)

    def test_return_tensors_functionality(self, embedder: Embedder):
        """Test return_tensors parameter."""
        utterances = ["Hello world", "Test sentence"]

        # Test numpy return (default)
        embeddings_np = embedder.embed(utterances, return_tensors=False)
        assert isinstance(embeddings_np, np.ndarray)

        # Test tensor return
        embeddings_tensor = embedder.embed(utterances, return_tensors=True)
        import torch

        assert isinstance(embeddings_tensor, torch.Tensor)

        # Values should be the same
        np.testing.assert_allclose(embeddings_np, embeddings_tensor.cpu().numpy(), rtol=1e-5)
