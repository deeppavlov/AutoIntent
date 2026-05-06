"""vLLM-based embedding backend for GPU-accelerated inference."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from autointent._hash import Hasher
from autointent._utils import require
from autointent.configs._embedder import VllmEmbeddingConfig

from .base import BaseEmbeddingBackend
from .utils import get_embeddings_path

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    from vllm import LLM

    from autointent.configs import TaskTypeEnum

logger = logging.getLogger(__name__)


class VllmEmbeddingBackend(BaseEmbeddingBackend):
    """vLLM-based embedding backend implementation."""

    supports_training: bool = False

    def __init__(self, config: VllmEmbeddingConfig) -> None:
        """Initialize the vLLM backend.

        Args:
            config: Configuration for vLLM embeddings.
        """
        self.config = config
        self._model = None

    def _load_model(self) -> LLM:
        """Lazy-load the vLLM LLM engine on first use."""
        if self._model is None:
            require("vllm", extra="vllm")
            from vllm import LLM

            kwargs = {
                "model": self.config.model_name,
                "task": "embed",
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "dtype": self.config.dtype,
                "trust_remote_code": self.config.trust_remote_code,
                **self.config.extra_init_kwargs,
            }
            if self.config.max_model_len is not None:
                kwargs["max_model_len"] = self.config.max_model_len

            logger.debug("Loading vLLM embedding model %s", self.config.model_name)
            self._model = LLM(**kwargs)
        return self._model

    def clear_ram(self) -> None:
        """Release GPU memory held by the vLLM engine."""
        if self._model is not None:
            logger.debug("Clearing vLLM embedder %s from GPU memory", self.config.model_name)
            del self._model
            self._model = None
            torch.cuda.empty_cache()

    def get_hash(self) -> int:
        """Compute a hash value for identifying the embedding model."""
        hasher = Hasher()
        hasher.update(self.config.model_name)
        hasher.update(str(self.config.max_model_len))
        return hasher.intdigest()

    def embed(
        self,
        utterances: list[str],
        task_type: TaskTypeEnum | None = None,
        return_tensors: bool = False,
    ) -> npt.NDArray[np.float32] | torch.Tensor:
        """Calculate embeddings for a list of utterances.

        Args:
            utterances: List of input texts to calculate embeddings for.
            task_type: Type of task for which embeddings are calculated.
            return_tensors: If True, return a PyTorch tensor; otherwise, return a numpy array.

        Returns:
            A numpy array or PyTorch tensor of embeddings.
        """
        if len(utterances) == 0:
            msg = "Empty input"
            logger.error(msg)
            raise ValueError(msg)

        prompt = self.config.get_prompt(task_type)
        if prompt:
            utterances = [f"{prompt} {utterance}" for utterance in utterances]

        if self.config.use_cache:
            hasher = Hasher()
            hasher.update(self.get_hash())
            hasher.update(utterances)
            if prompt:
                hasher.update(prompt)

            embeddings_path = get_embeddings_path(hasher.hexdigest())
            if embeddings_path.exists():
                logger.debug("Loading cached vLLM embeddings from %s", embeddings_path)
                embeddings_np = cast("npt.NDArray[np.float32]", np.load(embeddings_path))
                if return_tensors:
                    return torch.from_numpy(embeddings_np)
                return embeddings_np

        model = self._load_model()

        logger.debug(
            "Calculating embeddings with vLLM model %s, batch_size=%d",
            self.config.model_name,
            self.config.batch_size,
        )

        outputs = model.encode(utterances, pooling_task="embed", **self.config.extra_encode_kwargs)
        all_embeddings = [output.outputs.embedding for output in outputs]

        embeddings_np = np.array(all_embeddings, dtype=np.float32)

        if self.config.use_cache:
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, embeddings_np)

        if return_tensors:
            return torch.from_numpy(embeddings_np)
        return embeddings_np

    def similarity(
        self, embeddings1: npt.NDArray[np.float32], embeddings2: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Calculate cosine similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings (size n).
            embeddings2: Second set of embeddings (size m).

        Returns:
            A numpy array of similarities (size n x m).
        """
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        normalized1 = embeddings1 / norm1
        normalized2 = embeddings2 / norm2
        return cast("npt.NDArray[np.float32]", np.dot(normalized1, normalized2.T))

    def dump(self, path: Path) -> None:
        """Save the backend config to disk (stateless — no model weights to save).

        Args:
            path: Path to the directory where the backend will be saved.
        """
        path.mkdir(parents=True, exist_ok=True)
        config_path = path / "config.json"
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(self.config.model_dump(mode="json"), file, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> VllmEmbeddingBackend:
        """Load the backend from saved config.

        Args:
            path: Path to the directory where the backend is stored.

        Returns:
            Loaded backend instance.
        """
        config_path = path / "config.json"
        with config_path.open("r", encoding="utf-8") as file:
            config_data = json.load(file)
        config = VllmEmbeddingConfig.model_validate(config_data)
        return cls(config)
