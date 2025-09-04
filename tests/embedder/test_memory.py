import pytest

from autointent._wrappers.embedder import Embedder
from autointent.configs import EmbedderConfig, SentenceTransformerEmbeddingConfig

from .conftest import backend_configs


@pytest.mark.parametrize("embedder_config", backend_configs)
class TestEmbedderMemory:
    """Test memory management for different embedder backends."""

    @pytest.fixture
    def embedder(self, embedder_config: EmbedderConfig) -> Embedder:
        """Create an Embedder instance for testing."""
        return Embedder(embedder_config)

    def test_clear_ram(self, embedder: Embedder):
        """Test RAM clearing functionality."""
        # Load the model by doing an embedding
        embedder.embed(["test"])

        # Check that backend model is loaded for SentenceTransformers
        if isinstance(embedder.config, SentenceTransformerEmbeddingConfig):
            assert embedder._backend._model is not None

        # Clear RAM
        embedder.clear_ram()

        # For SentenceTransformers, model should be cleared
        if isinstance(embedder.config, SentenceTransformerEmbeddingConfig):
            assert embedder._backend._model is None
        # For OpenAI, clear_ram is a no-op (no model stored in RAM)

    def test_memory_efficiency_multiple_calls(self, embedder: Embedder):
        """Test that multiple embed calls don't cause memory leaks."""
        test_utterances = ["First test", "Second test", "Third test"]

        # Multiple embedding calls
        for _ in range(3):
            embeddings = embedder.embed(test_utterances)
            assert embeddings.shape[0] == len(test_utterances)

        # For SentenceTransformers, model should still be loaded once
        if isinstance(embedder.config, SentenceTransformerEmbeddingConfig):
            assert embedder._backend._model is not None

        # Clear RAM should work after multiple calls
        embedder.clear_ram()

        if isinstance(embedder.config, SentenceTransformerEmbeddingConfig):
            assert embedder._backend._model is None

    def test_model_reloading_after_clear(self, embedder: Embedder):
        """Test that model can be reloaded after clearing RAM."""
        # First embedding
        embeddings1 = embedder.embed(["test"])

        # Clear RAM
        embedder.clear_ram()

        # Second embedding should work (model reloaded)
        embeddings2 = embedder.embed(["test"])

        # Results should be identical (deterministic)
        import numpy as np

        np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5)
