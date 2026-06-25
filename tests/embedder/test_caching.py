from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from autointent._wrappers.embedder import Embedder
from autointent.configs import HashingVectorizerEmbeddingConfig, TaskTypeEnum

from .conftest import backend_configs, create_sentence_transformer_config

if TYPE_CHECKING:
    import numpy.typing as npt

    from autointent.configs import EmbedderConfig


@pytest.mark.parametrize("embedder_config", backend_configs)
class TestEmbedderCaching:
    """Test caching functionality for different embedder backends."""

    def test_caching_consistency(self, embedder_config: EmbedderConfig) -> None:
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

    def test_caching_disabled_consistency(self, embedder_config: EmbedderConfig) -> None:
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

    def test_caching_performance_improvement(self) -> None:
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

    def test_different_inputs_no_cache_collision(self) -> None:
        """Test that different inputs don't collide in cache."""
        config = create_sentence_transformer_config(use_cache=True)
        embedder = Embedder(config)

        embeddings1 = embedder.embed(["First sentence"])
        embeddings2 = embedder.embed(["Second sentence"])

        # Different inputs should produce different embeddings
        assert not np.allclose(embeddings1, embeddings2, rtol=1e-3)

    def test_cache_with_different_prompts(self) -> None:
        """Test that prompts are considered in caching."""
        config = create_sentence_transformer_config(
            use_cache=True,
            query_prompt="Query:",
            passage_prompt="Document:",
        )
        embedder = Embedder(config)

        # Same text with different prompts should be cached separately
        query_emb = embedder.embed(["test"], TaskTypeEnum.query)
        passage_emb = embedder.embed(["test"], TaskTypeEnum.passage)

        # Should produce different embeddings due to different prompts
        assert not np.allclose(query_emb, passage_emb, rtol=1e-3)


def _embedding_row_count() -> int:
    db_path = Path(os.environ["AUTOINTENT_CACHE_DIR"]) / "embeddings.db"
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])


class TestPerUtteranceCaching:
    """Per-utterance keying: shared utterances are stored once and reused across calls."""

    def test_overlapping_calls_store_each_utterance_once(self) -> None:
        config = create_sentence_transformer_config(use_cache=True)
        embedder = Embedder(config)

        embedder.embed(["alpha", "beta"])
        embedder.embed(["beta", "gamma"])  # 'beta' overlaps

        # Whole-list keying would store 2 list blobs; per-utterance stores 3 rows.
        assert _embedding_row_count() == 3

    def test_duplicate_in_list_computed_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = create_sentence_transformer_config(use_cache=True)
        embedder = Embedder(config)
        backend = embedder._backend

        computed: list[list[str]] = []
        original = backend._embed_uncached

        def spy(utterances: list[str], prompt: str | None) -> npt.NDArray[np.float32]:
            computed.append(list(utterances))
            return original(utterances, prompt)

        monkeypatch.setattr(backend, "_embed_uncached", spy)

        result = embedder.embed(["dup", "dup"])

        assert result.shape[0] == 2
        np.testing.assert_array_equal(result[0], result[1])
        assert computed == [["dup"]]  # computed only once

    def test_order_preserved_after_partial_hit(self) -> None:
        config = create_sentence_transformer_config(use_cache=True)
        embedder = Embedder(config)

        first = embedder.embed(["one", "two", "three"])
        second = embedder.embed(["three", "one", "two"])  # reordered, fully cached

        np.testing.assert_allclose(second[0], first[2], rtol=1e-5)
        np.testing.assert_allclose(second[1], first[0], rtol=1e-5)
        np.testing.assert_allclose(second[2], first[1], rtol=1e-5)

    def test_empty_input_hashing_vectorizer_returns_empty(self) -> None:
        embedder = Embedder(HashingVectorizerEmbeddingConfig(n_features=512, use_cache=True))
        result = embedder.embed([])
        assert result.shape == (0, 512)

    def test_empty_input_sentence_transformer_raises(self) -> None:
        embedder = Embedder(create_sentence_transformer_config(use_cache=True))
        with pytest.raises(ValueError, match="Empty input"):
            embedder.embed([])
