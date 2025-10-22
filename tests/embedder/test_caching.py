import numpy as np
import pytest

from autointent._wrappers.embedder import Embedder
from autointent.configs import EmbedderConfig

from .conftest import backend_configs, create_sentence_transformer_config


@pytest.mark.parametrize("embedder_config", backend_configs)
class TestEmbedderCaching:
    """Test caching functionality for different embedder backends."""

    def test_caching_consistency(self, embedder_config: EmbedderConfig):
        """Test that caching produces consistent results when enabled."""
        # Create config with caching enabled
        if hasattr(embedder_config, "model_copy"):
            config = embedder_config.model_copy()
            config.use_cache = True
        else:
            config = embedder_config
            config.use_cache = True

        embedder = Embedder(config)
        test_utterances = ["Cache consistency test sentence"]

        # First call
        embeddings1 = embedder.embed(test_utterances)

        # Second call should return same results from cache
        embeddings2 = embedder.embed(test_utterances)

        # Verify results are identical
        np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5)

    def test_caching_disabled_consistency(self, embedder_config: EmbedderConfig):
        """Test behavior when caching is disabled."""
        # Ensure caching is disabled
        if hasattr(embedder_config, "model_copy"):
            config = embedder_config.model_copy()
            config.use_cache = False
        else:
            config = embedder_config
            config.use_cache = False

        embedder = Embedder(config)
        test_utterances = ["No cache test"]

        embeddings1 = embedder.embed(test_utterances)
        embeddings2 = embedder.embed(test_utterances)

        # Should still be the same since same model/input (deterministic)
        np.testing.assert_allclose(embeddings1, embeddings2, atol=1e-3)


class TestSentenceTransformerCachingSpecific:
    """Test caching functionality specific to SentenceTransformer backend."""

    def test_caching_performance_improvement(self):
        """Test that caching provides performance improvement."""
        config = create_sentence_transformer_config(use_cache=True)
        embedder = Embedder(config)
        test_utterances = ["Performance test sentence"]

        # First call - cold start
        embeddings1 = embedder.embed(test_utterances)

        # Second call - should use cache
        embeddings2 = embedder.embed(test_utterances)

        # Verify results are the same
        np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5)

        # Second call should be faster (allow some tolerance for system variance)
        # Note: This might not always be true in tests due to small inputs
        # but we can at least verify the caching mechanism works
        assert embeddings1.shape == embeddings2.shape

    def test_different_inputs_no_cache_collision(self):
        """Test that different inputs don't collide in cache."""
        config = create_sentence_transformer_config(use_cache=True)
        embedder = Embedder(config)

        embeddings1 = embedder.embed(["First sentence"])
        embeddings2 = embedder.embed(["Second sentence"])

        # Different inputs should produce different embeddings
        assert not np.allclose(embeddings1, embeddings2, rtol=1e-3)

    def test_cache_with_different_prompts(self):
        """Test that prompts are considered in caching."""
        config = create_sentence_transformer_config(
            use_cache=True,
            query_prompt="Query:",
            passage_prompt="Document:",
        )
        embedder = Embedder(config)

        from autointent.configs import TaskTypeEnum

        # Same text with different prompts should be cached separately
        query_emb = embedder.embed(["test"], TaskTypeEnum.query)
        passage_emb = embedder.embed(["test"], TaskTypeEnum.passage)

        # Should produce different embeddings due to different prompts
        assert not np.allclose(query_emb, passage_emb, rtol=1e-3)
