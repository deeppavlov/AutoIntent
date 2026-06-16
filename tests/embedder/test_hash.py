from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from autointent import Embedder
from autointent.configs import SentenceTransformerEmbeddingConfig, TokenizerConfig

from .conftest import backend_configs

if TYPE_CHECKING:
    from autointent.configs import EmbedderConfig


@pytest.mark.parametrize("embedder_config", backend_configs)
class TestEmbedderHash:
    """Test hash generation for different embedder backends."""

    @pytest.fixture
    def embedder(self, embedder_config: EmbedderConfig) -> Embedder:
        """Create an Embedder instance for testing."""
        return Embedder(embedder_config)

    def test_hash_consistency(self, embedder: Embedder) -> None:
        """Test that hash generation is consistent for same configuration."""
        # Create second embedder with same config
        embedder2 = Embedder(embedder.config.model_copy(deep=True))

        # Same configuration should produce same hash
        assert embedder._get_hash() == embedder2._get_hash()

    def test_hash_deterministic(self, embedder: Embedder) -> None:
        """Test that hash is deterministic across multiple calls."""
        hash1 = embedder._get_hash()
        hash2 = embedder._get_hash()
        hash3 = embedder._get_hash()

        # Multiple calls should return same hash
        assert hash1 == hash2 == hash3


class TestSentenceTransformerHashSpecific:
    """Test hash generation specific to SentenceTransformer backend."""

    def test_hash_different_for_different_max_length(self) -> None:
        """Test that different max_length produces different hashes."""
        config1 = SentenceTransformerEmbeddingConfig(
            model_name="sergeyzh/rubert-tiny-turbo", tokenizer_config=TokenizerConfig(max_length=128)
        )
        config2 = SentenceTransformerEmbeddingConfig(
            model_name="sergeyzh/rubert-tiny-turbo", tokenizer_config=TokenizerConfig(max_length=256)
        )

        embedder1 = Embedder(config1)
        embedder2 = Embedder(config2)

        # Different max_length should produce different hashes
        assert embedder1._get_hash() != embedder2._get_hash()

    def test_hash_different_for_different_models(self) -> None:
        """Test that different models produce different hashes."""
        config1 = SentenceTransformerEmbeddingConfig(model_name="sergeyzh/rubert-tiny-turbo")
        config2 = SentenceTransformerEmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")

        embedder1 = Embedder(config1)
        embedder2 = Embedder(config2)

        # Different models should produce different hashes
        assert embedder1._get_hash() != embedder2._get_hash()
