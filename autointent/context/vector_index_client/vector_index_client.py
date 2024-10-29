import json
import logging
import shutil
from pathlib import Path
from typing import Any

from autointent.custom_types import LABEL_TYPE

from .cache import get_db_dir
from .vector_index import VectorIndex

DIRNAMES_TYPE = dict[str, str]


class VectorIndexClient:
    model_name: str

    def __init__(self, device: str, db_dir: str,         embedder_batch_size: int = 1,
        embedder_max_length: int | None = None,
**kwargs: dict[str, Any]) -> None:  # noqa: ARG002
        self._logger = logging.getLogger(__name__)
        self.device = device
        self.db_dir = get_db_dir(db_dir)
        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_length = embedder_max_length

    def create_index(
        self, model_name: str, utterances: list[str], labels: list[LABEL_TYPE]
    ) -> VectorIndex:
        """
        model_name should be a repo from hugging face, not a path to a local model
        """
        self._logger.info("Creating index for model: %s", model_name)

        index = VectorIndex(model_name, self.device, self.embedder_batch_size, self.embedder_max_length)
        index.add(utterances, labels)
        index.dump(self._get_dump_dirpath(model_name))

        self._add_index_dirname(model_name, str(self.db_dir))

        return index

    def _add_index_dirname(self, model_name: str, dir_name: str) -> None:
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
        """remove and return dirname if vector index exists, otherwise return None"""
        path = self.db_dir / "indexes_dirnames.json"
        with path.open() as file:
            indexes_dirnames: DIRNAMES_TYPE = json.load(file)
        dir_name = indexes_dirnames.pop(model_name, None)
        with path.open("w") as file:
            json.dump(indexes_dirnames, file, indent=4)
        return dir_name

    def _get_index_dirpath(self, model_name: str) -> Path | None:
        """return dirname if vector index exists, otherwise return None"""
        path = self.db_dir / "indexes_dirnames.json"
        with path.open() as file:
            indexes_dirnames: DIRNAMES_TYPE = json.load(file)
        dirname = indexes_dirnames.get(model_name, None)
        if dirname is None:
            return None
        return self.db_dir / dirname

    def _get_dump_dirpath(self, model_name: str) -> Path:
        if not self.db_dir.exists():
            self.db_dir.mkdir(parents=True, exist_ok=True)
        dir_name = model_name.replace("/", "-")
        self._add_index_dirname(model_name, dir_name)
        return self.db_dir / dir_name

    def delete_index(self, model_name: str) -> None:
        dir_name = self._remove_index_dirname(model_name)
        if dir_name is not None:
            self._logger.debug("Deleting index for model: %s", model_name)
            shutil.rmtree(self.db_dir / dir_name)

    def get_index(self, model_name: str) -> VectorIndex:
        dirpath = self._get_index_dirpath(model_name)
        if dirpath is not None:
            index = VectorIndex(model_name, self.device, self.embedder_batch_size, self.embedder_max_length)
            index.load(dirpath)
            return index

        msg = f"Index for {model_name} wasn't ever created in {dirpath}"
        self._logger.error(msg)
        raise NonExistentIndexError(msg)

    def get_or_create_index(self, model_name: str) -> VectorIndex:
        try:
            res = self.get_index(model_name)
        except NonExistentIndexError:
            res = self.create_index(model_name)
        return res


class NonExistentIndexError(Exception):
    def __init__(self, message: str = "non-existent index was requested") -> None:
        self.message = message
        super().__init__(message)
