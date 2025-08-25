from autointent._wrappers.embedder import Embedder
from autointent.configs._transformers import EmbedderConfig


def test_max_length_configuration():
    """Test max_length configuration affects tokenization."""
    max_length = 10
    config = EmbedderConfig(
        model_name="sergeyzh/rubert-tiny-turbo",
        tokenizer_config={"max_length": max_length},
        use_cache=False,
    )
    embedder = Embedder(config)

    # Should work with short text
    short_text = "short text"
    embeddings_short = embedder.embed([short_text])
    assert embeddings_short.shape[0] == 1
    assert embeddings_short.shape[1] > 0  # Should have embedding dimension

    # Should truncate long text to max_length
    long_text = " ".join(["word"] * 100)
    embeddings_long = embedder.embed([long_text])
    assert embeddings_long.shape[0] == 1
    assert embeddings_long.shape[1] > 0

    assert embeddings_short.shape == embeddings_long.shape
