from autointent import Embedder
from autointent.configs import EmbedderConfig, TokenizerConfig


def test_hash_consistency():
    """Test that hash generation is consistent for same configuration."""
    config1 = EmbedderConfig(model_name="sergeyzh/rubert-tiny-turbo")
    config2 = EmbedderConfig(model_name="sergeyzh/rubert-tiny-turbo")

    embedder1 = Embedder(config1)
    embedder2 = Embedder(config2)

    # Same configuration should produce same hash
    assert embedder1._get_hash() == embedder2._get_hash()


def test_hash_different_for_different_configs():
    """Test that different configurations produce different hashes."""
    config1 = EmbedderConfig(model_name="sergeyzh/rubert-tiny-turbo", tokenizer_config=TokenizerConfig(max_length=12))
    config2 = EmbedderConfig(model_name="sergeyzh/rubert-tiny-turbo", tokenizer_config=TokenizerConfig(max_length=13))

    embedder1 = Embedder(config1)
    embedder2 = Embedder(config2)

    # Different configurations should produce different hashes
    assert embedder1._get_hash() != embedder2._get_hash()
