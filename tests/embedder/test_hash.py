from __future__ import annotations

from typing import TYPE_CHECKING

import huggingface_hub
import huggingface_hub.constants
import pytest

from autointent import Embedder
from autointent._wrappers.embedder.sentence_transformers import _get_latest_commit_hash
from autointent.configs import SentenceTransformerEmbeddingConfig, TokenizerConfig

from .conftest import backend_configs

if TYPE_CHECKING:
    from pathlib import Path

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


class TestOfflineEmbedderCacheKey:
    """Regression tests for offline embedding cache key correctness (issue #321)."""

    def test_no_cross_model_collision_offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two different models with the same non-SHA revision must not collide when offline.

        Pre-fix: both fall back to the same revision string ("main") -> identical hashes.
        Post-fix: model_name is included in the hash -> distinct hashes.
        """
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
        _get_latest_commit_hash.cache_clear()

        config1 = SentenceTransformerEmbeddingConfig(model_name="org-a/model-alpha", revision="main", use_cache=False)
        config2 = SentenceTransformerEmbeddingConfig(model_name="org-b/model-beta", revision="main", use_cache=False)

        from autointent._wrappers.embedder.sentence_transformers import SentenceTransformerEmbeddingBackend

        backend1 = SentenceTransformerEmbeddingBackend(config1)
        backend2 = SentenceTransformerEmbeddingBackend(config2)

        assert backend1.get_hash() != backend2.get_hash()

        _get_latest_commit_hash.cache_clear()

    def test_offline_local_ref_resolution(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When offline and the HF cache contains refs/main, return the cached SHA without network.

        Verifies that _get_latest_commit_hash reads from the local ref file instead of
        calling model_info (which would raise under offline mode).
        """
        fake_sha = "a" * 40
        model_name = "some-org/some-model"

        from huggingface_hub.file_download import repo_folder_name

        repo_folder = repo_folder_name(repo_id=model_name, repo_type="model")
        ref_file = tmp_path / repo_folder / "refs" / "main"
        ref_file.parent.mkdir(parents=True)
        ref_file.write_text(fake_sha)

        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)

        # model_info must NOT be called — raise if it is
        def _no_network(*args: object, **kwargs: object) -> None:
            msg = "model_info called despite HF_HUB_OFFLINE=True"
            raise AssertionError(msg)

        monkeypatch.setattr(huggingface_hub, "model_info", _no_network)
        _get_latest_commit_hash.cache_clear()

        result = _get_latest_commit_hash(model_name, "main")
        assert result == fake_sha

        _get_latest_commit_hash.cache_clear()

    def test_offline_missing_ref_returns_revision(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When offline and no cached ref file exists, return the revision string without raising."""
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)

        def _no_network(*args: object, **kwargs: object) -> None:
            msg = "model_info called despite HF_HUB_OFFLINE=True"
            raise AssertionError(msg)

        monkeypatch.setattr(huggingface_hub, "model_info", _no_network)
        _get_latest_commit_hash.cache_clear()

        result = _get_latest_commit_hash("no-org/no-model", "main")
        assert result == "main"

        _get_latest_commit_hash.cache_clear()
