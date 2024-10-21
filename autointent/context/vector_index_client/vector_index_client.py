import json
import logging
import shutil
from pathlib import Path

from autointent.context.data_handler import DataHandler

from .vector_index import VectorIndex

DIRNAMES_TYPE = dict[str, str]


class VectorIndexClient:
    model_name: str

    def __init__(self, device: str, db_dir: str) -> None:
        self._logger = logging.getLogger(__name__)
        self.device = device
        self.db_dir = Path(db_dir)

    def create_index(self, model_name: str, data_handler: DataHandler) -> VectorIndex:
        """
        model_name should be a repo from hugging face, not a path to a local model
        """
        self._logger.info("Creating index for model: %s", model_name)

        index = VectorIndex(model_name, self.device)
        index.add(data_handler.utterances_train, data_handler.labels_train)

        index.dump(self._get_dump_dirpath(model_name))

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
            index = VectorIndex(model_name, self.device)
            index.load(dirpath)
            return index

        msg = f"index for {model_name} wasn't ever createds"
        self._logger.error(msg)
        raise ValueError(msg)
