import os
import platform

import pytest

from autointent.configs import OpenaiEmbeddingConfig, SentenceTransformerEmbeddingConfig

# Check if OpenAI API key is available for testing
openai_available = os.getenv("OPENAI_API_KEY") is not None

pytest.importorskip("sentence_transformers")


@pytest.fixture
def on_windows() -> bool:
    return platform.system() == "Windows"


# Backend configurations for parametrization
backend_configs = [
    pytest.param(
        SentenceTransformerEmbeddingConfig(
            model_name="sergeyzh/rubert-tiny-turbo",
            batch_size=4,
            device="cpu",
            use_cache=False,
        ),
        id="sentence_transformers",
    ),
    pytest.param(
        OpenaiEmbeddingConfig(
            model_name="text-embedding-3-small",
            batch_size=2,
            use_cache=False,
            max_retries=1,
            timeout=10.0,
        ),
        marks=pytest.mark.skipif(
            not openai_available,
            reason="OpenAI API key not available (set OPENAI_API_KEY environment variable)",
        ),
        id="openai",
    ),
]

# Only SentenceTransformer backend supports training
trainable_backend_configs = [
    pytest.param(
        SentenceTransformerEmbeddingConfig(
            model_name="sergeyzh/rubert-tiny-turbo",
            batch_size=4,
            device="cpu",
            use_cache=False,
        ),
        id="sentence_transformers_trainable",
    ),
]


def create_sentence_transformer_config(**kwargs) -> SentenceTransformerEmbeddingConfig:
    """Helper function to create SentenceTransformer config with defaults."""
    defaults = {
        "model_name": "sergeyzh/rubert-tiny-turbo",
        "batch_size": 4,
        "device": "cpu",
        "use_cache": False,
        "similarity_fn_name": "cosine",
    }
    defaults.update(kwargs)
    return SentenceTransformerEmbeddingConfig(**defaults)


def create_openai_config(**kwargs) -> OpenaiEmbeddingConfig:
    """Helper function to create OpenAI config with defaults."""
    defaults = {
        "model_name": "text-embedding-3-small",
        "batch_size": 2,
        "use_cache": False,
        "max_retries": 1,
        "timeout": 10.0,
    }
    defaults.update(kwargs)
    return OpenaiEmbeddingConfig(**defaults)
