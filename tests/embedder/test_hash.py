from autointent._wrappers.embedder import Embedder
from autointent.configs._transformers import EmbedderConfig


def test_hash_consistency():
    """Test that hash generation is consistent for same configuration."""
    config1 = EmbedderConfig(model_name="sergeyzh/rubert-tiny-turbo", batch_size=4)
    config2 = EmbedderConfig(model_name="sergeyzh/rubert-tiny-turbo", batch_size=4)

    embedder1 = Embedder(config1)
    embedder2 = Embedder(config2)

    # Same configuration should produce same hash
    assert embedder1._get_hash() == embedder2._get_hash()


def test_hash_different_for_different_configs():
    """Test that different configurations produce different hashes."""
    config1 = EmbedderConfig(model_name="sergeyzh/rubert-tiny-turbo", batch_size=4)
    config2 = EmbedderConfig(model_name="sergeyzh/rubert-tiny-turbo", batch_size=8)

    embedder1 = Embedder(config1)
    embedder2 = Embedder(config2)

    # Different configurations should produce different hashes
    assert embedder1._get_hash() != embedder2._get_hash()
