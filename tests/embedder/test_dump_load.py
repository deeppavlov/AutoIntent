import tempfile
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from autointent._wrappers.embedder import Embedder
from autointent.configs._transformers import EmbedderConfig


def test_load_from_disk(on_windows):
    model = SentenceTransformer("sergeyzh/rubert-tiny-turbo")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=on_windows) as tmp_dir:
        model.save(str(Path(tmp_dir) / "weights"))
        embedder = Embedder(EmbedderConfig(model_name=str(Path(tmp_dir) / "weights")))
        predictions = embedder.embed(["hi!"])
        embedder.dump(Path(tmp_dir) / "embedder")
        embedder_loaded = Embedder.load(Path(tmp_dir) / "embedder")
        predictions_after = embedder_loaded.embed(["hi!"])

    np.testing.assert_almost_equal(predictions_after, predictions, decimal=4)


def test_dump_load_cycle(on_windows):
    """Test complete dump/load cycle preserves functionality."""
    original_config = EmbedderConfig(
        model_name="sergeyzh/rubert-tiny-turbo",
        default_prompt="Test prompt:",
        similarity_fn_name="cosine",
        batch_size=4,
        use_cache=False,
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=on_windows) as temp_dir:
        temp_path = Path(temp_dir)

        # Create and test original embedder
        embedder_original = Embedder(original_config)
        test_utterances = ["Test sentence for persistence"]
        original_embeddings = embedder_original.embed(test_utterances)

        # Dump embedder
        embedder_original.dump(temp_path)

        # Load embedder
        embedder_loaded = Embedder.load(temp_path)

        # Test that loaded embedder works the same
        loaded_embeddings = embedder_loaded.embed(test_utterances)
        np.testing.assert_allclose(original_embeddings, loaded_embeddings, rtol=1e-5)

        # Test configuration preservation
        assert embedder_loaded.config.model_name == original_config.model_name
        assert embedder_loaded.config.default_prompt == original_config.default_prompt
        assert embedder_loaded.config.similarity_fn_name == original_config.similarity_fn_name


def test_load_with_config_override(on_windows):
    """Test loading with configuration override."""
    original_config = EmbedderConfig(
        model_name="sergeyzh/rubert-tiny-turbo",
        batch_size=8,
        use_cache=False,
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=on_windows) as temp_dir:
        temp_path = Path(temp_dir)

        # Create and dump original
        embedder_original = Embedder(original_config)
        embedder_original.dump(temp_path)

        # Load with override
        override_config = EmbedderConfig(batch_size=16)
        embedder_loaded = Embedder.load(temp_path, override_config)

        # Verify override took effect
        assert embedder_loaded.config.batch_size == 16
        # Verify original config preserved where not overridden
        assert embedder_loaded.config.model_name == original_config.model_name
