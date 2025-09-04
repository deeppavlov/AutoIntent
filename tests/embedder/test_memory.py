from autointent._wrappers.embedder import Embedder
from autointent.configs import SentenceTransformerEmbeddingConfig as EmbedderConfig


def test_clear_ram():
    """Test RAM clearing functionality."""
    config = EmbedderConfig(model_name="sergeyzh/rubert-tiny-turbo", use_cache=False)
    embedder = Embedder(config)

    embedder.embed(["test"])
    assert hasattr(embedder, "_model")

    embedder.clear_ram()

    assert not hasattr(embedder, "_model")
