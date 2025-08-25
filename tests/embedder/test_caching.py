import time
from unittest.mock import patch

import numpy as np

from autointent._wrappers.embedder import Embedder
from autointent.configs._transformers import EmbedderConfig


def test_caching_enabled():
    """Test that caching works when enabled."""
    config = EmbedderConfig(
        model_name="sergeyzh/rubert-tiny-turbo",
        use_cache=True,
        device="cpu",
    )
    embedder = Embedder(config)
    test_utterances = ["Cache test sentence"]

    # Mock the actual embedding calculation to verify caching
    with patch.object(embedder, "_load_model") as mock_load:
        mock_model = mock_load.return_value
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        # First call should trigger model loading
        start_time = time.time()
        embeddings1 = embedder.embed(test_utterances)
        first_call_time = time.time() - start_time

        # Second call should use cache (model.encode shouldn't be called again)
        start_time = time.time()
        embeddings2 = embedder.embed(test_utterances)
        second_call_time = time.time() - start_time

        # Verify results are the same
        np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5)

        assert (
            second_call_time < first_call_time / 5
        ), f"Second call ({second_call_time:.4f}s) should be much faster than first call ({first_call_time:.4f}s)"


def test_caching_disabled():
    """Test behavior when caching is disabled."""
    config = EmbedderConfig(
        model_name="sergeyzh/rubert-tiny-turbo",
        use_cache=False,
        device="cpu",
    )
    embedder = Embedder(config)
    test_utterances = ["No cache test"]

    embeddings1 = embedder.embed(test_utterances)
    embeddings2 = embedder.embed(test_utterances)

    # Should still be the same since same model/input
    np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5)
