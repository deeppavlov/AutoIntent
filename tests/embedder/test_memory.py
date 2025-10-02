import tempfile
from pathlib import Path

from autointent._wrappers.embedder import Embedder
from autointent.configs._transformers import EmbedderConfig


def test_clear_ram():
    """Test RAM clearing functionality."""
    config = EmbedderConfig(model_name="sergeyzh/rubert-tiny-turbo", use_cache=False)
    embedder = Embedder(config)

    embedder.embed(["test"])
    assert hasattr(embedder, "_model")

    embedder.clear_ram()

    assert not hasattr(embedder, "_model")


def test_delete_cleanup():
    """Test delete method cleans up resources."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        config = EmbedderConfig(model_name="sergeyzh/rubert-tiny-turbo", use_cache=False)
        embedder = Embedder(config)
        embedder.embed(["test"])  # Load model
        embedder.dump(temp_path)

        # Delete should clean up
        embedder.delete()

        # Directory should be removed
        assert not temp_path.exists()
