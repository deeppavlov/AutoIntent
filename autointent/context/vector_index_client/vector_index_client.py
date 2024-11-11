import json
import logging
import shutil
from pathlib import Path

from autointent.custom_types import LabelType

from .cache import get_db_dir
from .vector_index import VectorIndex

DIRNAMES_TYPE = dict[str, str]


class VectorIndexClient:
    def __init__(
        self,
        device: str,
        db_dir: str | Path | None,
        embedder_batch_size: int = 32,
        embedder_max_length: int | None = None,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self.device = device
        self.db_dir = get_db_dir(db_dir)
        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_length = embedder_max_length

    def create_index(
        self, model_name: str, utterances: list[str] | None = None, labels: list[LabelType] | None = None
    ) -> VectorIndex:
        """
        model_name should be a repo from hugging face, not a path to a local model
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
        index.dump(self._get_dump_dirpath(index.model_name))

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
        if not path.exists():
            return None
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

        msg = f"Index for {model_name} wasn't ever created in {self.db_dir}"
        self._logger.error(msg)
        raise NonExistingIndexError(msg)

    def exists(self, model_name: str) -> bool:
        return self._get_index_dirpath(model_name) is not None

    def delete_db(self) -> None:
        shutil.rmtree(self.db_dir)


class NonExistingIndexError(Exception):
    def __init__(self, message: str = "non-existent index was requested") -> None:
        self.message = message
        super().__init__(message)
