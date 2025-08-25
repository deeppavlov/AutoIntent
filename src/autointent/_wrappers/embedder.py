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
from uuid import uuid4

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
from transformers import EarlyStoppingCallback, TrainerCallback

from autointent._hash import Hasher
from autointent.configs import EmbedderConfig, EmbedderFineTuningConfig, TaskTypeEnum
from autointent.custom_types import ListOfLabels

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


class Embedder:
    """A wrapper for managing embedding models using :py:class:`sentence_transformers.SentenceTransformer`.

    This class handles initialization, saving, loading, and clearing of
    embedding models, as well as calculating embeddings for input texts.
    """

    _metadata_dict_name: str = "metadata.json"
    _weights_dir_name: str = "sentence_transformer"
    _dump_dir: Path | None = None
    _trained: bool = False
    _model: SentenceTransformer

    def __init__(self, embedder_config: EmbedderConfig) -> None:
        """Initialize the Embedder.

        Args:
            embedder_config: Config of embedder.
        """
        self.config = embedder_config.model_copy(deep=True)

    def _get_hash(self) -> int:
        """Compute a hash value for the Embedder.

        Returns:
            The hash value of the Embedder.
        """
        hasher = Hasher()
        if not Path(self.config.model_name).exists():
            commit_hash = _get_latest_commit_hash(self.config.model_name)
            hasher.update(commit_hash)
        else:
            self._model = self._load_model()
            for parameter in self._model.parameters():
                hasher.update(parameter.detach().cpu().numpy())
        hasher.update(self.config.tokenizer_config.max_length)
        return hasher.intdigest()

    def _load_model(self) -> SentenceTransformer:
        """Load sentence transformers model to device."""
        if not hasattr(self, "_model"):
            res = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                prompts=self.config.get_prompt_config(),
                similarity_fn_name=self.config.similarity_fn_name,
                trust_remote_code=self.config.trust_remote_code,
            )
        else:
            res = self._model
        return res

    def train(self, utterances: list[str], labels: ListOfLabels, config: EmbedderFineTuningConfig) -> None:
        """Train the embedding model."""
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

        self._model = self._load_model()

        x_train, x_val, y_train, y_val = train_test_split(utterances, labels, test_size=config.val_fraction)
        tr_ds = Dataset.from_dict({"text": x_train, "label": y_train})
        val_ds = Dataset.from_dict({"text": x_val, "label": y_val})

        loss = BatchAllTripletLoss(model=self._model, margin=config.margin)
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SentenceTransformerTrainingArguments(
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
                batch_sampler=BatchSamplers.NO_DUPLICATES,
                metric_for_best_model="eval_loss",
                load_best_model_at_end=True,
                eval_strategy="epoch",
                greater_is_better=False,
            )
            callbacks: list[TrainerCallback] = [
                EarlyStoppingCallback(
                    early_stopping_patience=config.early_stopping_patience,
                    early_stopping_threshold=config.early_stopping_threshold,
                )
            ]
            trainer = SentenceTransformerTrainer(
                model=self._model,
                args=args,
                train_dataset=tr_ds,
                eval_dataset=val_ds,
                loss=loss,
                callbacks=callbacks,
            )

            trainer.train()

        # use temporary path for re-usage
        model_path = str(Path(tempfile.mkdtemp("autointent_embedders")) / str(uuid4()))
        self._model.save(model_path)
        self.config.model_name = model_path

        self._trained = True

    def clear_ram(self) -> None:
        """Move the embedding model to CPU and delete it from memory."""
        if hasattr(self, "_model"):
            logger.debug("Clearing embedder %s from memory", self.config.model_name)
            self._model.cpu()
            del self._model
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
        if self._trained:
            model_path = str((path / self._weights_dir_name).resolve())
            self._model.save(model_path, create_model_card=False)
            self.config.model_name = model_path

        self._dump_dir = path
        path.mkdir(parents=True, exist_ok=True)
        with (path / self._metadata_dict_name).open("w") as file:
            json.dump(self.config.model_dump(mode="json"), file, indent=4)

    @classmethod
    def load(cls, path: Path | str, override_config: EmbedderConfig | None = None) -> "Embedder":
        """Load the embedding model and metadata from disk.

        Args:
            path: Path to the directory where the model is stored.
            override_config: one can override presaved settings
        """
        with (Path(path) / cls._metadata_dict_name).open(encoding="utf-8") as file:
            config = EmbedderConfig.model_validate_json(file.read())

        if override_config is not None:
            kwargs = {**config.model_dump(), **override_config.model_dump(exclude_unset=True)}
        else:
            kwargs = config.model_dump()

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
        if len(utterances) == 0:
            msg = "Empty input"
            logger.error(msg)
            raise ValueError(msg)

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

        self._model = self._load_model()

        logger.debug(
            "Calculating embeddings with model %s, batch_size=%d, max_seq_length=%s, embedder_device=%s, prompt=%s",
            self.config.model_name,
            self.config.batch_size,
            str(self.config.tokenizer_config.max_length),
            self.config.device,
            prompt,
        )

        if self.config.tokenizer_config.max_length is not None:
            self._model.max_seq_length = self.config.tokenizer_config.max_length

        embeddings = self._model.encode(
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
