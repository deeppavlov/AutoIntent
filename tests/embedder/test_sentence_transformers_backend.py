"""Tests specific to SentenceTransformer backend functionality."""

import pytest

from autointent._wrappers.embedder.sentence_transformers import SentenceTransformerEmbeddingBackend
from autointent.configs import EmbedderFineTuningConfig, SentenceTransformerEmbeddingConfig


@pytest.fixture
def st_backend_config() -> SentenceTransformerEmbeddingConfig:
    """Create a SentenceTransformer backend config for testing."""
    return SentenceTransformerEmbeddingConfig(
        model_name="sergeyzh/rubert-tiny-turbo",
        batch_size=4,
        device="cpu",
        use_cache=False,
        similarity_fn_name="cosine",
    )


@pytest.fixture
def st_backend(st_backend_config: SentenceTransformerEmbeddingConfig) -> SentenceTransformerEmbeddingBackend:
    """Create a SentenceTransformer backend instance."""
    return SentenceTransformerEmbeddingBackend(st_backend_config)


class TestSentenceTransformerBackend:
    """Test SentenceTransformer-specific backend functionality."""

    def test_backend_initialization(self, st_backend: SentenceTransformerEmbeddingBackend) -> None:
        """Test backend initialization."""
        assert st_backend.supports_training is True
        assert st_backend._model is None  # Model should be lazy-loaded
        assert st_backend._trained is False

    def test_model_lazy_loading(self, st_backend: SentenceTransformerEmbeddingBackend) -> None:
        """Test that model is lazy-loaded."""
        assert st_backend._model is None

        # Model should be loaded on first embed call
        embeddings = st_backend.embed(["Test sentence"])
        # reason: mypy narrowed `_model` to `None` from the prior assert and
        # cannot see the mutation inside `.embed()`. The post-call assert is
        # the whole point of this test (lazy load: None -> non-None).
        assert st_backend._model is not None
        assert embeddings.shape == (1, st_backend._model.get_sentence_embedding_dimension())  # type: ignore[unreachable]

    def test_clear_ram(self, st_backend: SentenceTransformerEmbeddingBackend) -> None:
        """Test clearing model from RAM."""
        # Load model
        st_backend.embed(["Test sentence"])
        assert st_backend._model is not None

        # Clear RAM
        st_backend.clear_ram()
        assert st_backend._model is None

    def test_similarity_function_name(self, st_backend: SentenceTransformerEmbeddingBackend) -> None:
        """Test that similarity function is configured correctly."""
        embeddings = st_backend.embed(["Hello", "World"])
        similarity = st_backend.similarity(embeddings[:1], embeddings[1:])

        # Should return cosine similarity
        assert similarity.shape == (1, 1)
        assert -1.0 <= similarity[0, 0] <= 1.0

    def test_hash_calculation(self, st_backend: SentenceTransformerEmbeddingBackend) -> None:
        """Test hash calculation for caching."""
        hash1 = st_backend.get_hash()
        hash2 = st_backend.get_hash()

        # Hash should be deterministic
        assert hash1 == hash2
        assert isinstance(hash1, int)

    def test_training_functionality(self, st_backend: SentenceTransformerEmbeddingBackend) -> None:
        """Test basic training functionality."""
        pytest.importorskip("accelerate", reason="Accelerate library is required for this test")

        # Simple training data
        utterances = [
            "Hello world",
            "Good morning",
            "Hello there",
            "Good evening",
            "Hi there",
            "Good night",
        ]
        labels = [0, 1, 0, 1, 0, 1]  # Two classes

        # Training config with minimal epochs for testing
        train_config = EmbedderFineTuningConfig(epoch_num=1, batch_size=2)

        # Get original hash
        original_hash = st_backend.get_hash()

        # Train the model
        st_backend.train(utterances, labels, train_config)

        # Check that training state is updated
        assert st_backend._trained is True

        # Hash should change after training (model path changes)
        new_hash = st_backend.get_hash()
        assert new_hash != original_hash
