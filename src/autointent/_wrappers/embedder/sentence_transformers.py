from __future__ import annotations

import json
import logging
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, overload
from uuid import uuid4

import huggingface_hub
import numpy as np
import numpy.typing as npt
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

from autointent._hash import Hasher
from autointent._utils import require
from autointent.configs._embedder import SentenceTransformerEmbeddingConfig

from .base import BaseEmbeddingBackend
from .utils import get_embeddings_path

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers import TrainerCallback

    from autointent.configs import EmbedderFineTuningConfig, TaskTypeEnum
    from autointent.custom_types import ListOfLabels

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _get_latest_commit_hash(model_name: str) -> str:
    """Get the latest commit hash for a given Hugging Face model.

    Args:
        model_name: The name of the model to get the latest commit hash for.

    Returns:
        The latest commit hash for the given model name or the model name if the commit hash is not found.
    """
    commit_hash = huggingface_hub.model_info(model_name, revision="main").sha
    if commit_hash is None:
        logger.warning("No commit hash found for model %s", model_name)
        return model_name
    return commit_hash


class SentenceTransformerEmbeddingBackend(BaseEmbeddingBackend):
    """SentenceTransformer-based embedding backend implementation."""

    supports_training: bool = True
    _model: SentenceTransformer | None

    def __init__(self, config: SentenceTransformerEmbeddingConfig) -> None:
        """Initialize the SentenceTransformer backend.

        Args:
            config: Configuration for SentenceTransformer embeddings.
        """
        self.config = config
        self._model = None
        self._trained: bool = False

    def clear_ram(self) -> None:
        """Move the embedding model to CPU and delete it from memory."""
        if self._model is not None:
            logger.debug("Clearing embedder %s from memory", self.config.model_name)
            self._model.cpu()
            del self._model
            self._model = None
            torch.cuda.empty_cache()

    def _load_model(self) -> SentenceTransformer:
        """Load sentence transformers model to device."""
        if self._model is None:
            # Lazy import sentence-transformers
            st = require("sentence_transformers", extra="sentence-transformers")
            res = st.SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                prompts=self.config.get_prompt_config(),
                similarity_fn_name=self.config.similarity_fn_name,
                trust_remote_code=self.config.trust_remote_code,
            )
            self._model = res
        return self._model

    def get_hash(self) -> int:
        """Compute a hash value for the backend.

        Returns:
            The hash value of the backend.
        """
        hasher = Hasher()
        if not Path(self.config.model_name).exists():
            commit_hash = _get_latest_commit_hash(self.config.model_name)
            hasher.update(commit_hash)
        else:
            model = self._load_model()
            for parameter in model.parameters():
                hasher.update(parameter.detach().cpu().numpy())
        hasher.update(self.config.tokenizer_config.max_length)
        return hasher.intdigest()

    @overload
    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, *, return_tensors: Literal[True]
    ) -> torch.Tensor: ...

    @overload
    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, *, return_tensors: Literal[False] = False
    ) -> npt.NDArray[np.float32]: ...

    def embed(
        self, utterances: list[str], task_type: TaskTypeEnum | None = None, return_tensors: bool = False
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

        if self.config.use_cache:
            logger.debug("Using cached embeddings for %s", self.config.model_name)
            hasher = Hasher()
            hasher.update(self.get_hash())
            hasher.update(utterances)
            if prompt:
                hasher.update(prompt)

            embeddings_path = get_embeddings_path(hasher.hexdigest())
            if embeddings_path.exists():
                logger.debug("loading embeddings from %s", str(embeddings_path))
                embeddings_np = cast("npt.NDArray[np.float32]", np.load(embeddings_path))
                if return_tensors:
                    device = self.config.device or "cpu"
                    return torch.from_numpy(embeddings_np).to(device)
                return embeddings_np

        model = self._load_model()

        logger.debug(
            "Calculating embeddings with model %s, batch_size=%d, max_seq_length=%s, embedder_device=%s, prompt=%s",
            self.config.model_name,
            self.config.batch_size,
            str(self.config.tokenizer_config.max_length),
            self.config.device,
            prompt,
        )

        if self.config.tokenizer_config.max_length is not None:
            model.max_seq_length = self.config.tokenizer_config.max_length

        embeddings: npt.NDArray[np.float32] | torch.Tensor
        if return_tensors:
            embeddings = model.encode(
                utterances,
                convert_to_tensor=True,
                batch_size=self.config.batch_size,
                normalize_embeddings=True,
                prompt=prompt,
            )
        else:
            embeddings = cast(
                "npt.NDArray[np.float32]",
                model.encode(
                    utterances,
                    convert_to_numpy=True,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=True,
                    prompt=prompt,
                ),
            )

        if self.config.use_cache:
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(embeddings, torch.Tensor):
                np.save(embeddings_path, embeddings.cpu().numpy())
            else:
                np.save(embeddings_path, embeddings)

        return embeddings

    def similarity(
        self, embeddings1: npt.NDArray[np.float32], embeddings2: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Calculate similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings (size n).
            embeddings2: Second set of embeddings (size m).

        Returns:
            A numpy array of similarities (size n x m).
        """
        model = self._load_model()
        return model.similarity(embeddings1, embeddings2).detach().cpu().numpy().astype(np.float32)

    def train(self, utterances: list[str], labels: ListOfLabels, config: EmbedderFineTuningConfig) -> None:
        """Train the embedding model.

        Args:
            utterances: List of training utterances.
            labels: List of labels corresponding to utterances.
            config: Fine-tuning configuration.
        """
        if len(utterances) != len(labels):
            msg = f"Utterances and labels lists lengths mismatch: {len(utterances)=} != {len(labels)=}"
            raise ValueError(msg)

        if len(labels) == 0:
            msg = "Empty data"
            raise ValueError(msg)

        # TODO support multi-label data
        if isinstance(labels[0], list):
            msg = "Multi-label data is not supported for embeddings fine-tuning for now"
            logger.warning(msg)
            return

        model = self._load_model()

        # Lazy import sentence-transformers training components (only needed for fine-tuning)
        st = require("sentence_transformers", extra="sentence-transformers")
        transformers = require("transformers", extra="transformers")

        x_train, x_val, y_train, y_val = train_test_split(utterances, labels, test_size=config.val_fraction)
        tr_ds = Dataset.from_dict({"text": x_train, "label": y_train})
        val_ds = Dataset.from_dict({"text": x_val, "label": y_val})

        loss = st.losses.BatchAllTripletLoss(model=model, margin=config.margin)
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = st.SentenceTransformerTrainingArguments(
                save_strategy="epoch",
                save_total_limit=1,
                output_dir=tmp_dir,
                num_train_epochs=config.epoch_num,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                warmup_ratio=config.warmup_ratio,
                fp16=config.fp16,
                bf16=config.bf16,
                batch_sampler=st.training_args.BatchSamplers.NO_DUPLICATES,
                metric_for_best_model="eval_loss",
                load_best_model_at_end=True,
                eval_strategy="epoch",
                greater_is_better=False,
            )
            callbacks: list[TrainerCallback] = [
                transformers.EarlyStoppingCallback(
                    early_stopping_patience=config.early_stopping_patience,
                    early_stopping_threshold=config.early_stopping_threshold,
                )
            ]
            trainer = st.SentenceTransformerTrainer(
                model=model,
                args=args,
                train_dataset=tr_ds,
                eval_dataset=val_ds,
                loss=loss,
                callbacks=callbacks,
            )

            trainer.train()

        # use temporary path for re-usage
        model_path = str(Path(tempfile.mkdtemp("autointent_embedders")) / str(uuid4()))
        model.save(model_path)
        self.config.model_name = model_path

        self._trained = True

    def dump(self, path: Path) -> None:
        """Save the backend state to disk.

        Args:
            path: Path to the directory where the backend will be saved.
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save the configuration
        config_path = path / "config.json"
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(self.config.model_dump(mode="json"), file, indent=4, ensure_ascii=False)

        # Save trained model if exists
        if self._trained and hasattr(self, "_model") and self._model is not None:
            model_path = path / "sentence_transformer"
            self._model.save(str(model_path), create_model_card=False)

            # Save training state
            training_state_path = path / "training_state.json"
            with training_state_path.open("w", encoding="utf-8") as file:
                json.dump({"trained": True}, file, indent=4)

    @classmethod
    def load(cls, path: Path) -> SentenceTransformerEmbeddingBackend:
        """Load the backend state from disk.

        Args:
            path: Path to the directory where the backend is stored.

        Returns:
            Loaded backend instance.
        """
        # Load configuration
        config_path = path / "config.json"
        with config_path.open("r", encoding="utf-8") as file:
            config_data = json.load(file)

        config = SentenceTransformerEmbeddingConfig.model_validate(config_data)

        # Check if a trained model exists
        model_path = path / "sentence_transformer"
        training_state_path = path / "training_state.json"

        if model_path.exists() and training_state_path.exists():
            # Update config to point to the saved model
            config.model_name = str(model_path)

        # Create instance
        instance = cls(config)

        # Set training state if applicable
        if training_state_path.exists():
            with training_state_path.open("r", encoding="utf-8") as file:
                training_state = json.load(file)
            instance._trained = training_state.get("trained", False)

        return instance
