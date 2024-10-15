import logging
import shutil
from pathlib import Path

from autointent.context.data_handler import DataHandler

from .vector_index import VectorIndex


class VectorIndexClient:
    def __init__(self, device: str, multilabel: bool, n_classes: int, db_dir: str) -> None:
        self._logger = logging.getLogger(__name__)
        self.device = device
        self.multilabel = multilabel
        self.n_classes = n_classes
        self.db_dir = Path(db_dir)
        self.indexes_dirnames: dict[str, str] = {}
        self.indexes_alive: dict[str, VectorIndex] = {}

    def create_index(self, model_name: str, data_handler: DataHandler) -> VectorIndex:
        self._logger.info("Creating index for model: %s", model_name)

        index = VectorIndex(model_name, self.device)
        index.add(data_handler.utterances_train, data_handler.labels_train)

        self.indexes_alive[model_name] = index
        index.dump(self.get_dump_dir(model_name))

        return index

    def get_dump_dir(self, model_name: str) -> Path:
        dir_name = model_name.replace("/", "-")
        self.indexes_dirnames[model_name] = dir_name
        return self.db_dir / dir_name

    def delete_index(self, model_name: str) -> None:
        if model_name in self.indexes_alive:
            self._logger.debug("Killing index for model: %s", model_name)
            index = self.indexes_alive.pop(model_name)
            index.delete()

        if model_name in self.indexes_dirnames:
            self._logger.debug("Deleting index for model: %s", model_name)
            dir_name = self.indexes_dirnames.pop(model_name)
            shutil.rmtree(self.db_dir / dir_name)

    def get_index(self, model_name: str) -> VectorIndex:
        if model_name in self.indexes_dirnames:
            index = VectorIndex(model_name, self.device)
            index.load(self.db_dir / self.indexes_dirnames[model_name])
            return index

        msg = f"index for {model_name} wasn't ever createds"
        self._logger.error(msg)
        raise ValueError(msg)
