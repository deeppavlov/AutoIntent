import numpy as np
import pytest

from autointent._wrappers.embedder import Embedder
from autointent.configs import EmbedderConfig, TaskTypeEnum

from .conftest import backend_configs, create_openai_config, create_sentence_transformer_config


@pytest.mark.parametrize("embedder_config", backend_configs)
class TestEmbedderPrompts:
    """Test prompt functionality for different embedder backends."""

    @pytest.fixture
    def prompt_embedder_config(self, embedder_config: EmbedderConfig) -> EmbedderConfig:
        """Create embedder config with different prompts based on backend type."""
        from autointent.configs import HashingVectorizerEmbeddingConfig

        # Skip for HashingVectorizer as it doesn't support prompts
        if isinstance(embedder_config, HashingVectorizerEmbeddingConfig):
            pytest.skip("HashingVectorizer doesn't support prompts")

        if hasattr(embedder_config, "similarity_fn_name"):
            # SentenceTransformers config
            return create_sentence_transformer_config(
                default_prompt="Represent this text:",
                query_prompt="Search query:",
                passage_prompt="Document:",
                classification_prompt="Classify:",
                use_cache=False,
            )
        # OpenAI config
        return create_openai_config(
            default_prompt="Represent this text:",
            query_prompt="Search query:",
            passage_prompt="Document:",
            classification_prompt="Classify:",
            use_cache=False,
        )

    def test_different_task_prompts(self, prompt_embedder_config: EmbedderConfig):
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

    def test_fallback_to_default_prompt(self, embedder_config: EmbedderConfig):
        """Test fallback to default prompt when specific prompt not set."""
        from autointent.configs import HashingVectorizerEmbeddingConfig

        # Skip for HashingVectorizer as it doesn't support prompts
        if isinstance(embedder_config, HashingVectorizerEmbeddingConfig):
            pytest.skip("HashingVectorizer doesn't support prompts")

        if hasattr(embedder_config, "similarity_fn_name"):
            # SentenceTransformers config
            config = create_sentence_transformer_config(
                default_prompt="Default:",
                use_cache=False,
            )
        else:
            # OpenAI config
            config = create_openai_config(
                default_prompt="Default:",
                use_cache=False,
            )

        embedder = Embedder(config)

        # Should use default prompt when specific task prompt not available
        embeddings1 = embedder.embed(["test"], TaskTypeEnum.cluster)
        embeddings2 = embedder.embed(["test"], TaskTypeEnum.default)

        np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5)
