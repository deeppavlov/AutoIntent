"""Module for managing vector indexes using a client interface.

This module provides the `VectorIndexClient` class for creating, managing, and
persisting vector indexes that utilize FAISS and embedding models. It also
includes functionality to handle persistent storage and retrieval of indexes.
"""

import json
import logging
from pathlib import Path

from autointent.custom_types import LabelType

from .cache import get_db_dir
from .vector_index import VectorIndex

DIRNAMES_TYPE = dict[str, str]


class VectorIndexClient:
    """
    Client interface for managing vector indexes.

    This class provides methods for creating, persisting, loading, and deleting
    vector indexes. Indexes are stored in a specified directory and associated
    with embedding models.
    """

    def __init__(
        self,
        device: str,
        db_dir: str | Path | None,
        embedder_batch_size: int = 32,
        embedder_max_length: int | None = None,
    ) -> None:
        """
        Initialize the VectorIndexClient.

        :param device: Device to run the embedding model on.
        :param db_dir: Directory for storing vector indexes. Defaults to a cache directory.
        :param embedder_batch_size: Batch size for the embedding model.
        :param embedder_max_length: Maximum sequence length for the embedding model.
        """
        self._logger = logging.getLogger(__name__)
        self.device = device
        self.db_dir = get_db_dir(db_dir)
        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_length = embedder_max_length

    def create_index(
        self,
        model_name: str,
        utterances: list[str] | None = None,
        labels: list[LabelType] | None = None,
    ) -> VectorIndex:
        """
        Create a new vector index for the specified model.

        :param model_name: Name of the embedding model (Hugging Face repo, not a local path).
        :param utterances: Optional list of utterances to add to the index.
        :param labels: Optional list of labels corresponding to the utterances.
        :return: A `VectorIndex` instance.
        :raises ValueError: If only one of `utterances` or `labels` is provided.
        """
        self._logger.info("Creating index for model: %s", model_name)

        index = VectorIndex(model_name, self.device, self.embedder_batch_size, self.embedder_max_length)
        if utterances is not None and labels is not None:
            index.add(utterances, labels)
            self.dump(index)
        elif (utterances is not None) != (labels is not None):
            msg = "You must provide both utterances and labels, or neither"
            raise ValueError(msg)

        return index

    def dump(self, index: VectorIndex) -> None:
        """
        Persist the vector index to disk.

        :param index: The `VectorIndex` instance to save.
        """
        index.dump(self._get_dump_dirpath(index.model_name))

    def _add_index_dirname(self, model_name: str, dir_name: str) -> None:
        """
        Add a directory name to the index mapping.

        :param model_name: Name of the model.
        :param dir_name: Directory name associated with the model.
        """
        path = self.db_dir / "indexes_dirnames.json"
        if path.exists():
            with path.open() as file:
                indexes_dirnames: DIRNAMES_TYPE = json.load(file)
        else:
            indexes_dirnames = {}
        indexes_dirnames[model_name] = dir_name
        with path.open("w") as file:
            json.dump(indexes_dirnames, file, indent=4)

    def _remove_index_dirname(self, model_name: str) -> str | None:
        """
        Remove and return the directory name for a given model, if it exists.

        :param model_name: Name of the model.
        :return: The removed directory name, or None if not found.
        """
        path = self.db_dir / "indexes_dirnames.json"
        with path.open() as file:
            indexes_dirnames: DIRNAMES_TYPE = json.load(file)
        dir_name = indexes_dirnames.pop(model_name, None)
        with path.open("w") as file:
            json.dump(indexes_dirnames, file, indent=4)
        return dir_name

    def _get_index_dirpath(self, model_name: str) -> Path | None:
        """
        Retrieve the directory path for a given model, if it exists.

        :param model_name: Name of the model.
        :return: The directory path, or None if not found.
        """
        path = self.db_dir / "indexes_dirnames.json"
        if not path.exists():
            return None
        with path.open() as file:
            indexes_dirnames: DIRNAMES_TYPE = json.load(file)
        dirname = indexes_dirnames.get(model_name, None)
        if dirname is None:
            return None
        return self.db_dir / dirname

    def _get_dump_dirpath(self, model_name: str) -> Path:
        """
        Generate and retrieve the directory path for saving a model.

        :param model_name: Name of the model.
        :return: The directory path for saving the model.
        """
        if not self.db_dir.exists():
            self.db_dir.mkdir(parents=True, exist_ok=True)
        dir_name = model_name.replace("/", "-")
        self._add_index_dirname(model_name, dir_name)
        return self.db_dir / dir_name

    def delete_index(self, model_name: str) -> None:
        """
        Delete a vector index and its associated data.

        :param model_name: Name of the model.
        """
        if not self.exists(model_name):
            return
        index = self.get_index(model_name)
        index.delete()

    def get_index(self, model_name: str) -> VectorIndex:
        """
        Load a vector index for a given model.

        :param model_name: Name of the model.
        :return: The loaded `VectorIndex` instance.
        :raises NonExistingIndexError: If the index does not exist.
        """
        dirpath = self._get_index_dirpath(model_name)
        if dirpath is not None:
            index = VectorIndex(model_name, self.device, self.embedder_batch_size, self.embedder_max_length)
            index.load(dirpath)
            return index

        msg = f"Index for {model_name} wasn't ever created in {self.db_dir}"
        self._logger.error(msg)
        raise NonExistingIndexError(msg)

    def exists(self, model_name: str) -> bool:
        """
        Check if a vector index exists for a given model.

        :param model_name: Name of the model.
        :return: True if the index exists, False otherwise.
        """
        return self._get_index_dirpath(model_name) is not None

    def delete_db(self) -> None:
        """Delete all vector indexes and their associated data from disk."""
        path = self.db_dir / "indexes_dirnames.json"
        if not path.exists():
            return
        with path.open() as file:
            indexes_dirnames: DIRNAMES_TYPE = json.load(file)
        for embedder_name in indexes_dirnames:
            self.delete_index(embedder_name)
        path.unlink()


class NonExistingIndexError(Exception):
    """Exception raised when a non-existent vector index is requested."""

    def __init__(self, message: str = "Non-existent index was requested") -> None:
        """
        Initialize the exception.

        :param message: The error message.
        """
        self.message = message
        super().__init__(message)
