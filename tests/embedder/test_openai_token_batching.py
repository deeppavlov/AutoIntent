"""Token-aware batching for OpenAI embeddings (no API calls)."""

import pytest

pytest.importorskip("openai")
tiktoken = pytest.importorskip("tiktoken")

from autointent._wrappers.embedder.openai import (  # noqa: E402
    OpenaiEmbeddingBackend,
    _batch_strings_by_token_budget,
)
from autointent.configs import OpenaiEmbeddingConfig  # noqa: E402


def test_batch_strings_none_max_tokens_uses_batch_size_only() -> None:
    texts = list("abcdef")
    batches = _batch_strings_by_token_budget(
        texts,
        model_name="text-embedding-3-small",
        max_strings_per_batch=3,
        max_tokens_per_batch=None,
    )
    assert batches == [["a", "b", "c"], ["d", "e", "f"]]


def test_batch_strings_respects_token_budget() -> None:
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    batches = _batch_strings_by_token_budget(
        ["hello"] * 25,
        model_name="text-embedding-3-small",
        max_strings_per_batch=100,
        max_tokens_per_batch=10,
    )
    assert sum(len(b) for b in batches) == 25
    for batch in batches:
        total = sum(len(encoding.encode(t)) for t in batch)
        assert total <= 10


def test_batch_strings_respects_string_count_and_tokens() -> None:
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    batches = _batch_strings_by_token_budget(
        ["hello"] * 6,
        model_name="text-embedding-3-small",
        max_strings_per_batch=2,
        max_tokens_per_batch=100,
    )
    assert len(batches) == 3
    assert all(len(b) <= 2 for b in batches)
    for batch in batches:
        assert sum(len(encoding.encode(t)) for t in batch) <= 100


def test_embedding_request_batches_on_backend() -> None:
    config = OpenaiEmbeddingConfig(
        model_name="text-embedding-3-small",
        batch_size=100,
        max_tokens_in_batch=10,
        use_cache=False,
    )
    backend = OpenaiEmbeddingBackend(config)
    batches = backend._embedding_request_batches(["hello"] * 12)
    assert sum(len(b) for b in batches) == 12
