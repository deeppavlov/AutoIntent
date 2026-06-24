"""Drive the REAL OpenaiEmbeddingBackend against an httpx-level (respx) mock.

These tests import the backend class directly, so the autouse fake in
``conftest.py`` (which only rebinds the symbol inside ``embedder.py``) does not
apply. They exercise the real client construction, batching, sync/async paths
and error handling without any network access. They require the ``openai``
extra and therefore run in the embedder CI job.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import numpy as np
import pytest
import torch

pytest.importorskip("openai")

from autointent._wrappers.embedder.openai import OpenaiEmbeddingBackend
from autointent.configs import OpenaiEmbeddingConfig

if TYPE_CHECKING:
    from respx.router import MockRouter

EMBED_DIM = 3


def _embeddings_response(request: httpx.Request) -> httpx.Response:
    """Return one fixed-size embedding per input string in the request."""
    payload = json.loads(request.content)
    inputs = payload["input"]
    count = len(inputs) if isinstance(inputs, list) else 1
    data = [{"object": "embedding", "index": i, "embedding": [0.1] * EMBED_DIM} for i in range(count)]
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": data,
            "model": payload["model"],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )


def _config(**kwargs: object) -> OpenaiEmbeddingConfig:
    defaults = {"model_name": "text-embedding-3-small", "batch_size": 2, "use_cache": False, "max_retries": 0}
    return OpenaiEmbeddingConfig(**{**defaults, **kwargs})  # type: ignore[arg-type]


def test_embed_sync_returns_array(respx_openai: MockRouter) -> None:
    route = respx_openai.post("/v1/embeddings").mock(side_effect=_embeddings_response)
    backend = OpenaiEmbeddingBackend(_config())

    result = backend.embed(["hello", "world", "foo"])

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, EMBED_DIM)
    assert route.called


def test_embed_async_returns_array(respx_openai: MockRouter) -> None:
    route = respx_openai.post("/v1/embeddings").mock(side_effect=_embeddings_response)
    backend = OpenaiEmbeddingBackend(_config(max_concurrent=1))

    result = backend.embed(["hello", "world"])

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, EMBED_DIM)
    assert route.called


def test_embed_return_tensors(respx_openai: MockRouter) -> None:
    respx_openai.post("/v1/embeddings").mock(side_effect=_embeddings_response)
    backend = OpenaiEmbeddingBackend(_config())

    result = backend.embed(["hello"], return_tensors=True)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, EMBED_DIM)


def test_embed_empty_raises() -> None:
    backend = OpenaiEmbeddingBackend(_config())
    with pytest.raises(ValueError, match="Empty input"):
        backend.embed([])


def test_embed_api_error_wrapped_in_runtime_error(respx_openai: MockRouter) -> None:
    respx_openai.post("/v1/embeddings").mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))
    backend = OpenaiEmbeddingBackend(_config())

    with pytest.raises(RuntimeError):
        backend.embed(["hello"])
