"""Deterministic seeded fake of OpenaiEmbeddingBackend for tests."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import pytest
import torch

from autointent._wrappers.embedder.base import BaseEmbeddingBackend

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from autointent.configs import OpenaiEmbeddingConfig, TaskTypeEnum


def _seeded_vector(text: str, dim: int, *, seed_extra: str = "") -> npt.NDArray[np.float32]:
    """Deterministic unit vector from text+extra seed.

    Uses sha256 to seed numpy's RNG, then normalises so cosine similarity is well-defined.
    Vectors derived from textually-similar inputs (e.g. shared prefixes "hello world" / "hello")
    tend to be closer than unrelated inputs because the digest commutes through
    a fixed prefix when we mix in token-level hashes (see _mix_token_hashes).
    """
    digest = hashlib.sha256(f"{seed_extra}|{text}".encode()).digest()
    base_seed = int.from_bytes(digest[:8], "little") % (2**32)
    rng = np.random.default_rng(base_seed)
    raw = rng.standard_normal(dim).astype(np.float32)

    # Mix in per-token hashes so shared tokens nudge vectors closer.
    for token in text.lower().split():
        token_seed = int.from_bytes(hashlib.sha256(f"{seed_extra}|{token}".encode()).digest()[:8], "little") % (2**32)
        token_rng = np.random.default_rng(token_seed)
        raw += 0.5 * token_rng.standard_normal(dim).astype(np.float32)

    norm = np.linalg.norm(raw)
    return raw / norm if norm > 0 else raw


class FakeOpenaiEmbeddingBackend(BaseEmbeddingBackend):
    """In-process stand-in for OpenaiEmbeddingBackend.

    Mirrors the public surface that tests touch:
      - supports_training = False
      - lazy _client / _async_client attributes (kept for the test_client_lazy_loading test)
      - embed(...) returning (n, dim) array; dim = config.dimensions or 1536
      - similarity(...) cosine
      - get_hash() stable per-config

    Network is never touched.
    """

    supports_training = False

    def __init__(self, config: OpenaiEmbeddingConfig) -> None:
        self.config = config
        # Mirror the lazy-client attributes the real backend has so existing tests work.
        self._client = None
        self._async_client = None

    def clear_ram(self) -> None:
        self._client = None
        self._async_client = None

    @overload
    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, *, return_tensors: Literal[True]
    ) -> torch.Tensor: ...

    @overload
    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, *, return_tensors: Literal[False] = False
    ) -> npt.NDArray[np.float32]: ...

    def embed(
        self,
        utterances: list[str],
        task_type: TaskTypeEnum | None = None,
        return_tensors: bool = False,
    ) -> npt.NDArray[np.float32] | torch.Tensor:
        # Touch the lazy attributes so test_client_lazy_loading observes the transition.
        self._client = self._client or object()
        dim = getattr(self.config, "dimensions", None) or 1536

        # Prompt seed mirrors BaseEmbedderConfig.get_prompt() so that two task types
        # sharing the same default_prompt produce identical vectors.
        prompt = self.config.get_prompt(task_type)
        seed_extra = f"{self.config.model_name}|{prompt or ''}"

        vectors = np.stack([_seeded_vector(text, dim, seed_extra=seed_extra) for text in utterances])
        if return_tensors:
            return torch.from_numpy(vectors)
        return vectors

    def similarity(
        self, embeddings1: npt.NDArray[np.float32], embeddings2: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        # Inputs are already unit-normalised; cosine = dot product.
        return embeddings1 @ embeddings2.T

    def get_hash(self) -> int:
        # Stable hash from model_name + dimensions; matches real backend semantics.
        payload = json.dumps(
            {
                "model_name": self.config.model_name,
                "dimensions": getattr(self.config, "dimensions", None),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode()).digest()
        return int.from_bytes(digest[:8], "little")

    def dump(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "fake_openai_backend.json").write_text(
            json.dumps(
                {"model_name": self.config.model_name, "dimensions": getattr(self.config, "dimensions", None)},
                sort_keys=True,
            )
        )

    @classmethod
    def load(cls, path: Path) -> FakeOpenaiEmbeddingBackend:
        from autointent.configs import OpenaiEmbeddingConfig

        data = json.loads((path / "fake_openai_backend.json").read_text())
        return cls(OpenaiEmbeddingConfig(model_name=data["model_name"], dimensions=data.get("dimensions")))


@pytest.fixture
def patch_openai_embedding_backend(monkeypatch):
    """Rebind OpenaiEmbeddingBackend inside Embedder so the factory builds the fake.

    Verified call sites (src/autointent/_wrappers/embedder/embedder.py):
      - line 23: `from .openai import OpenaiEmbeddingBackend`
      - line 66: `_init_backend()` -> `OpenaiEmbeddingBackend(self.config)`
      - line 163: `load()` -> `OpenaiEmbeddingBackend.load(backend_path)`

    Both call sites resolve the symbol against `embedder.py`'s module namespace,
    so a single setattr on that module covers construction AND load.
    """
    from autointent._wrappers.embedder import embedder as embedder_module

    monkeypatch.setattr(embedder_module, "OpenaiEmbeddingBackend", FakeOpenaiEmbeddingBackend)
