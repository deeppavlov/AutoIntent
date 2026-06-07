import importlib.util
import platform

import pytest
import torch

from autointent.configs import (
    HashingVectorizerEmbeddingConfig,
    OpenaiEmbeddingConfig,
    SentenceTransformerEmbeddingConfig,
    VllmEmbeddingConfig,
)

# Check if vLLM is installed and CUDA is available
vllm_available = importlib.util.find_spec("vllm") is not None and torch.cuda.is_available()


@pytest.fixture
def on_windows() -> bool:
    return platform.system() == "Windows"


# Backend configurations for parametrization
backend_configs = [
    pytest.param(
        HashingVectorizerEmbeddingConfig(
            n_features=512,
            use_cache=False,
        ),
        id="hashing_vectorizer",
    ),
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
        id="openai",
    ),
    pytest.param(
        VllmEmbeddingConfig(
            model_name="sergeyzh/rubert-tiny-turbo",
            batch_size=4,
            use_cache=False,
            max_model_len=512,
        ),
        marks=pytest.mark.skipif(
            not vllm_available,
            reason="vLLM not installed or CUDA not available (pip install autointent[vllm])",
        ),
        id="vllm",
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


def create_vllm_config(**kwargs) -> VllmEmbeddingConfig:
    """Helper function to create VllmEmbeddingConfig with test-friendly defaults."""
    defaults = {
        "model_name": "BAAI/bge-base-en-v1.5",
        "batch_size": 4,
        "use_cache": False,
        "gpu_memory_utilization": 0.5,
        "max_model_len": 512,
    }
    defaults.update(kwargs)
    return VllmEmbeddingConfig(**defaults)


@pytest.fixture(autouse=True)
def _autouse_fake_openai_embedding(patch_openai_embedding_backend):
    """Within tests/embedder/, every OpenaiEmbeddingConfig resolves to FakeOpenaiEmbeddingBackend."""
