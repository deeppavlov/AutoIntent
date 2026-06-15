from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from autointent._wrappers.embedder import Embedder
from autointent.configs import (
    OpenaiEmbeddingConfig,
    SentenceTransformerEmbeddingConfig,
    VllmEmbeddingConfig,
)
from tests.conftest import tiny_sentence_transformer

from .conftest import backend_configs

if TYPE_CHECKING:
    from autointent.configs import EmbedderConfig


def test_load_from_disk(on_windows: bool) -> None:
    """Test loading embedder from disk with custom saved model."""
    model = tiny_sentence_transformer()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=on_windows) as tmp_dir:
        model.save(str(Path(tmp_dir) / "weights"))
        embedder = Embedder(SentenceTransformerEmbeddingConfig(model_name=str(Path(tmp_dir) / "weights")))
        predictions = embedder.embed(["hi!"])
        embedder.dump(Path(tmp_dir) / "embedder")
        embedder_loaded = Embedder.load(Path(tmp_dir) / "embedder")
        predictions_after = embedder_loaded.embed(["hi!"])

    np.testing.assert_almost_equal(predictions_after, predictions, decimal=4)


@pytest.mark.parametrize("embedder_config", backend_configs)
class TestEmbedderDumpLoad:
    """Unified test class for Embedder dump/load with different backends."""

    @pytest.fixture
    def embedder(self, embedder_config: EmbedderConfig) -> Embedder:
        """Create an Embedder instance for testing."""
        return Embedder(embedder_config)

    def test_dump_load_cycle(
        self,
        embedder: Embedder,
        on_windows: bool,
        embedder_config: EmbedderConfig,  # noqa: ARG002
    ) -> None:
        """Test complete dump/load cycle preserves functionality."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=on_windows) as temp_dir:
            temp_path = Path(temp_dir)

            # Create and test original embedder
            test_utterances = ["Test sentence for persistence", "Another test sentence"]
            original_embeddings = embedder.embed(test_utterances)

            # Dump embedder
            embedder.dump(temp_path)

            # Load embedder
            embedder_loaded = Embedder.load(temp_path)

            # Test that loaded embedder works the same
            loaded_embeddings = embedder_loaded.embed(test_utterances)
            np.testing.assert_allclose(original_embeddings, loaded_embeddings, rtol=1e-3)

            # Test configuration preservation (only for configs that have these attributes).
            # The BaseEmbedderConfig union doesn't expose backend-specific fields; assert
            # the concrete subclass(es) to narrow before attribute access.
            named_configs = (
                SentenceTransformerEmbeddingConfig,
                OpenaiEmbeddingConfig,
                VllmEmbeddingConfig,
            )
            if hasattr(embedder.config, "model_name"):
                assert isinstance(embedder_loaded.config, named_configs)
                assert isinstance(embedder.config, named_configs)
                assert embedder_loaded.config.model_name == embedder.config.model_name
            if hasattr(embedder.config, "default_prompt"):
                assert embedder_loaded.config.default_prompt == embedder.config.default_prompt
            if hasattr(embedder.config, "batch_size"):
                assert isinstance(embedder_loaded.config, named_configs)
                assert isinstance(embedder.config, named_configs)
                assert embedder_loaded.config.batch_size == embedder.config.batch_size

    def test_load_with_config_override(
        self,
        embedder: Embedder,
        on_windows: bool,
        embedder_config: EmbedderConfig,  # noqa: ARG002
    ) -> None:
        """Test loading with configuration override."""
        from autointent.configs import HashingVectorizerEmbeddingConfig

        # Skip for HashingVectorizer as it doesn't support batch_size override
        if isinstance(embedder.config, HashingVectorizerEmbeddingConfig):
            pytest.skip("HashingVectorizer doesn't support batch_size configuration")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=on_windows) as temp_dir:
            temp_path = Path(temp_dir)

            # Create and dump original
            embedder.dump(temp_path)

            # Create appropriate override config based on backend type
            override_config: EmbedderConfig
            if isinstance(embedder.config, SentenceTransformerEmbeddingConfig):
                override_config = SentenceTransformerEmbeddingConfig(batch_size=16)
            else:
                # For OpenAI, we can override batch_size too
                override_config = OpenaiEmbeddingConfig(batch_size=16)

            # Load with override
            embedder_loaded = Embedder.load(temp_path, override_config)

            # Verify override took effect. embedder_loaded.config is the union
            # BaseEmbedderConfig | ...; both SentenceTransformer and Openai
            # subclasses carry batch_size/model_name, so assert isinstance to narrow.
            assert isinstance(embedder_loaded.config, (SentenceTransformerEmbeddingConfig, OpenaiEmbeddingConfig))
            assert isinstance(embedder.config, (SentenceTransformerEmbeddingConfig, OpenaiEmbeddingConfig))
            assert embedder_loaded.config.batch_size == 16
            # Verify original config preserved where not overridden
            assert embedder_loaded.config.model_name == embedder.config.model_name

    def test_similarity_preserved_after_load(self, embedder: Embedder, on_windows: bool) -> None:
        """Test that similarity function works correctly after dump/load."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=on_windows) as temp_dir:
            temp_path = Path(temp_dir)

            # Test similarity with original embedder
            utterances = ["Hello world", "Test sentence"]
            embeddings = embedder.embed(utterances)
            original_similarity = embedder.similarity(embeddings[:1], embeddings[1:])

            # Dump and load
            embedder.dump(temp_path)
            embedder_loaded = Embedder.load(temp_path)

            # Test similarity with loaded embedder
            loaded_embeddings = embedder_loaded.embed(utterances)
            loaded_similarity = embedder_loaded.similarity(loaded_embeddings[:1], loaded_embeddings[1:])

            # Similarities should be the same
            np.testing.assert_allclose(original_similarity, loaded_similarity, rtol=1e-3)

    def test_multiple_dump_load_cycles(self, embedder: Embedder, on_windows: bool) -> None:
        """Test multiple dump/load cycles maintain consistency."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=on_windows) as temp_dir:
            temp_path = Path(temp_dir)
            test_utterances = ["Consistency test"]

            # Original embeddings
            original_embeddings = embedder.embed(test_utterances)

            # First dump/load cycle
            embedder.dump(temp_path / "cycle1")
            embedder_1 = Embedder.load(temp_path / "cycle1")
            embeddings_1 = embedder_1.embed(test_utterances)

            # Second dump/load cycle
            embedder_1.dump(temp_path / "cycle2")
            embedder_2 = Embedder.load(temp_path / "cycle2")
            embeddings_2 = embedder_2.embed(test_utterances)

            # All embeddings should be consistent
            np.testing.assert_allclose(original_embeddings, embeddings_1, atol=1e-3)
            np.testing.assert_allclose(embeddings_1, embeddings_2, atol=1e-3)
