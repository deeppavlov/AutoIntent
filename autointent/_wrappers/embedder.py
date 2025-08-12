"""Module for managing embedding models using Sentence Transformers.

This module provides the `Embedder` class for managing, persisting, and loading
embedding models and calculating embeddings for input texts.
"""

import json
import logging
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import huggingface_hub
import numpy as np
import numpy.typing as npt
import torch
from appdirs import user_cache_dir
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import BatchAllTripletLoss
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.training_args import BatchSamplers
from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback

from autointent._hash import Hasher
from autointent.configs import EmbedderConfig, EmbedderFineTuningConfig, TaskTypeEnum

logger = logging.getLogger(__name__)


def _get_embeddings_path(filename: str) -> Path:
    """Get the path to the embeddings file.

    This function constructs the full path to an embeddings file stored
    in a specific directory under the user's home directory. The embeddings
    file is named based on the provided filename, with the `.npy` extension
    added.

    Args:
        filename: The name of the embeddings file (without extension).

    Returns:
        The full path to the embeddings file.
    """
    return Path(user_cache_dir("autointent")) / "embeddings" / f"{filename}.npy"


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


class EmbedderDumpMetadata(TypedDict):
    """Metadata for saving and loading an Embedder instance."""

    model_name: str
    """Name of the hugging face model or a local path to sentence transformers dump."""
    device: str | None
    """Torch notation for CPU or CUDA."""
    batch_size: int
    """Batch size used for embedding calculations."""
    max_length: int | None
    """Maximum sequence length for the embedding model."""
    use_cache: bool
    """Whether to use embeddings caching."""
    similarity_fn_name: str | None
    """Name of the similarity function to use."""


class Embedder:
    """A wrapper for managing embedding models using :py:class:`sentence_transformers.SentenceTransformer`.

    This class handles initialization, saving, loading, and clearing of
    embedding models, as well as calculating embeddings for input texts.
    """

    _metadata_dict_name: str = "metadata.json"
    _dump_dir: Path | None = None
    embedding_model: SentenceTransformer

    def __init__(self, embedder_config: EmbedderConfig) -> None:
        """Initialize the Embedder.

        Args:
            embedder_config: Config of embedder.
        """
        self.config = embedder_config

    def _get_hash(self) -> int:
        """Compute a hash value for the Embedder.

        Returns:
            The hash value of the Embedder.
        """
        hasher = Hasher()
        if self.config.freeze:
            commit_hash = _get_latest_commit_hash(self.config.model_name)
            hasher.update(commit_hash)
        else:
            self._load_model()
            for parameter in self.embedding_model.parameters():
                hasher.update(parameter.detach().cpu().numpy())
        hasher.update(self.config.tokenizer_config.max_length)
        return hasher.intdigest()

    def _load_model(self) -> None:
        """Load sentence transformers model to device."""
        if not hasattr(self, "embedding_model"):
            self.embedding_model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                prompts=self.config.get_prompt_config(),
                similarity_fn_name=self.config.similarity_fn_name,
                trust_remote_code=self.config.trust_remote_code,
            )

    def train(self, utterances: list[str], labels: list[int], config: EmbedderFineTuningConfig) -> None:
        """Train the embedding model."""
        self._load_model()
        if config.early_stopping:
            x_train, x_val, y_train, y_val = train_test_split(utterances, labels, test_size=0.1, random_state=42)
            tr_ds = Dataset.from_dict({"text": x_train, "label": y_train})
            val_ds = Dataset.from_dict({"text": x_val, "label": y_val})
        else:
            tr_ds = Dataset.from_dict({"text": utterances, "label": labels})
            val_ds = None

        loss = BatchAllTripletLoss(model=self.embedding_model, margin=config.margin)
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SentenceTransformerTrainingArguments(
                save_strategy="no",
                output_dir=tmp_dir,
                num_train_epochs=config.epoch_num,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=8,
                eval_steps=1,
                learning_rate=config.learning_rate,
                warmup_ratio=config.warmup_ratio,
                fp16=config.fp16,
                bf16=config.bf16,
                batch_sampler=BatchSamplers.NO_DUPLICATES,
                metric_for_best_model="eval_loss",
                load_best_model_at_end=True,
                evaluation_strategy="epoch",
                greater_is_better=False,
            )
            callback = []
            if config.early_stopping:
                callback.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=config.early_stopping,
                        early_stopping_threshold=config.early_stopping_threshold,
                    )
                )
            trainer = SentenceTransformerTrainer(
                model=self.embedding_model,
                args=args,
                train_dataset=tr_ds,
                eval_dataset=val_ds,
                loss=loss,
                callbacks=callback,
            )

            trainer.train()

    def clear_ram(self) -> None:
        """Move the embedding model to CPU and delete it from memory."""
        if hasattr(self, "embedding_model"):
            logger.debug("Clearing embedder %s from memory", self.config.model_name)
            self.embedding_model.cpu()
            del self.embedding_model
            torch.cuda.empty_cache()

    def delete(self) -> None:
        """Delete the embedding model and its associated directory."""
        self.clear_ram()
        if self._dump_dir is not None:
            shutil.rmtree(self._dump_dir)

    def dump(self, path: Path) -> None:
        """Save the embedding model and metadata to disk.

        Args:
            path: Path to the directory where the model will be saved.
        """
        self._dump_dir = path
        metadata = EmbedderDumpMetadata(
            model_name=str(self.config.model_name),
            device=self.config.device,
            batch_size=self.config.batch_size,
            max_length=self.config.tokenizer_config.max_length,
            use_cache=self.config.use_cache,
            similarity_fn_name=self.config.similarity_fn_name,
        )
        path.mkdir(parents=True, exist_ok=True)
        with (path / self._metadata_dict_name).open("w") as file:
            json.dump(metadata, file, indent=4)

    @classmethod
    def load(cls, path: Path | str, override_config: EmbedderConfig | None = None) -> "Embedder":
        """Load the embedding model and metadata from disk.

        Args:
            path: Path to the directory where the model is stored.
            override_config: one can override presaved settings
        """
        with (Path(path) / cls._metadata_dict_name).open(encoding="utf-8") as file:
            metadata: EmbedderDumpMetadata = json.load(file)

        if override_config is not None:
            kwargs = {**metadata, **override_config.model_dump(exclude_unset=True)}
        else:
            kwargs = metadata  # type: ignore[assignment]

        max_length = kwargs.pop("max_length", None)
        if max_length is not None:
            kwargs["tokenizer_config"] = {"max_length": max_length}

        return cls(EmbedderConfig(**kwargs))

    def embed(self, utterances: list[str], task_type: TaskTypeEnum | None = None) -> npt.NDArray[np.float32]:
        """Calculate embeddings for a list of utterances.

        Args:
            utterances: List of input texts to calculate embeddings for.
            task_type: Type of task for which embeddings are calculated.

        Returns:
            A numpy array of embeddings.
        """
        prompt = self.config.get_prompt(task_type)

        if self.config.use_cache:
            logger.debug("Using cached embeddings for %s", self.config.model_name)
            hasher = Hasher()
            hasher.update(self._get_hash())
            hasher.update(utterances)
            if prompt:
                hasher.update(prompt)

            embeddings_path = _get_embeddings_path(hasher.hexdigest())
            if embeddings_path.exists():
                logger.debug("loading embeddings from %s", str(embeddings_path))
                return np.load(embeddings_path)  # type: ignore[no-any-return]

        self._load_model()

        logger.debug(
            "Calculating embeddings with model %s, batch_size=%d, max_seq_length=%s, embedder_device=%s, prompt=%s",
            self.config.model_name,
            self.config.batch_size,
            str(self.config.tokenizer_config.max_length),
            self.config.device,
            prompt,
        )

        if self.config.tokenizer_config.max_length is not None:
            self.embedding_model.max_seq_length = self.config.tokenizer_config.max_length

        embeddings = self.embedding_model.encode(
            utterances,
            convert_to_numpy=True,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            prompt=prompt,
        )

        if self.config.use_cache:
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
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
        similarity_fn = SimilarityFunction.to_similarity_fn(self.config.similarity_fn_name)
        return similarity_fn(embeddings1, embeddings2).detach().cpu().numpy().astype(np.float32)
